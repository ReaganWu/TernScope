"""TilePolypNet-V1: a Gemmini-oriented segmentation co-design candidate.

V1 is intentionally a new architecture rather than a small V0 patch.  Its
deployment graph avoids channel-wise depthwise convolution and softmax, keeps
all internal widths aligned to a 16x16 systolic array, and exposes fusion
regions which the compiler may lower without intermediate layout materializing.

The model is still an *untrained* candidate.  Shape/export/compiler checks are
useful now; segmentation quality must be established by training and ablation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .model import ConvBNAct


ARRAY_DIM = 16


def _relu6() -> nn.Module:
    return nn.ReLU6(inplace=False)


class RepChannelMixer(nn.Module):
    """A 1x1 Conv-BN plus identity which becomes one affine conv in deploy.

    The identity is folded into the diagonal of the pointwise weight.  This
    preserves the train-time residual while removing an activation read, add,
    and write from inference.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % ARRAY_DIM:
            raise ValueError("channels must be a multiple of 16")
        self.channels = channels
        self.conv: nn.Module = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn: nn.Module = nn.BatchNorm2d(channels)
        self.deploy = False

    def forward(self, x: Tensor) -> Tensor:
        y = self.bn(self.conv(x))
        return y if self.deploy else y + x

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        assert isinstance(self.conv, nn.Conv2d)
        assert isinstance(self.bn, nn.BatchNorm2d)
        fused = fuse_conv_bn_eval(self.conv, self.bn)
        identity = torch.eye(
            self.channels, device=fused.weight.device, dtype=fused.weight.dtype
        ).reshape(self.channels, self.channels, 1, 1)
        fused.weight.add_(identity)
        self.conv = fused
        self.bn = nn.Identity()
        self.deploy = True


class ArrayPartialSpatialBlock(nn.Module):
    """Spatially mix one array-aligned channel tile, then mix all channels.

    This adapts Partial Convolution to a 16-wide systolic array.  The untouched
    channel slice is a view in the compiler IR; ``cat -> RepChannelMixer`` is a
    declared fusion pattern and must not become a standalone memory copy.
    """

    def __init__(self, channels: int, partial_channels: int | None = None) -> None:
        super().__init__()
        if channels % ARRAY_DIM:
            raise ValueError("channels must be a multiple of 16")
        if partial_channels is None:
            partial_channels = max(ARRAY_DIM, math.ceil(channels / 4 / ARRAY_DIM) * ARRAY_DIM)
        if not 0 < partial_channels < channels or partial_channels % ARRAY_DIM:
            raise ValueError("partial_channels must be array aligned and smaller than channels")
        self.channels = channels
        self.partial_channels = partial_channels
        self.spatial = ConvBNAct(
            partial_channels, partial_channels, kernel_size=3, activation=True
        )
        self.channel_mixer = RepChannelMixer(channels)
        self.activation = _relu6()

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial(x[:, : self.partial_channels])
        bypass = x[:, self.partial_channels :]
        # The V1 compiler contracts this concat with the following pointwise
        # convolution and writes both logical slices directly into its im2col
        # descriptor.  Eager PyTorch still executes the obvious semantics.
        mixed = torch.cat((spatial, bypass), dim=1)
        return self.activation(self.channel_mixer(mixed))


class ArrayLinearContext(nn.Module):
    """ReLU linear attention without an N-by-N score tensor or softmax.

    Context is restricted to the two lowest-resolution encoder stages.  The
    reciprocal is one scalar per query/head and is assigned to a vector LUT;
    the two large products remain dense, array-friendly matmuls.
    """

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads or channels // heads % ARRAY_DIM:
            raise ValueError("per-head width must be a multiple of 16")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.out_proj = ConvBNAct(channels, channels, 1, activation=False)
        self.activation = _relu6()

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = F.relu(q).reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
        k = F.relu(k).reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
        v = v.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)

        # Associativity changes O(N^2 d) attention into O(N d^2).
        kv = torch.matmul(k.transpose(-2, -1), v)
        numerator = torch.matmul(q, kv)
        denominator = torch.matmul(q, k.sum(dim=-2).unsqueeze(-1)).clamp_min(1e-6)
        context = numerator / denominator
        context = context.transpose(-2, -1).reshape(batch, channels, height, width)
        return self.activation(x + self.out_proj(context))


@dataclass(frozen=True)
class ScheduleRegion:
    name: str
    backend: str
    contract: str
    materialized_output: bool = True


class TilePolypNetV1(nn.Module):
    """Array-first polyp segmentation candidate for 256x256 INT8 inference."""

    channel_plan: Tuple[int, ...] = (16, 32, 64, 128, 256)

    def __init__(self, num_classes: int = 1, decoder_channels: int = 64) -> None:
        super().__init__()
        if decoder_channels != 64:
            raise ValueError("V1 currently fixes decoder_channels=64 for its RTL contract")
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels

        # Only dense convolution is used for downsampling; no groups=C kernels.
        self.stem = nn.Sequential(
            ConvBNAct(3, 16, 3, stride=2),
            ConvBNAct(16, 32, 3, stride=2),
        )
        self.stage1 = nn.Sequential(
            ArrayPartialSpatialBlock(32, 16),
            ArrayPartialSpatialBlock(32, 16),
        )

        self.down2 = ConvBNAct(32, 64, 1, stride=2)
        self.stage2 = nn.Sequential(
            ArrayPartialSpatialBlock(64, 16),
            ArrayPartialSpatialBlock(64, 16),
        )

        self.down3 = ConvBNAct(64, 128, 1, stride=2)
        self.stage3 = nn.Sequential(
            ArrayPartialSpatialBlock(128, 32),
            ArrayPartialSpatialBlock(128, 32),
            ArrayLinearContext(128, heads=4),
        )

        self.down4 = ConvBNAct(128, 256, 1, stride=2)
        self.stage4 = nn.Sequential(
            ArrayPartialSpatialBlock(256, 64),
            ArrayLinearContext(256, heads=8),
        )

        # Parallel high-level partial decoder: x2/x3/x4 are projected
        # independently, then accumulated directly into the x2-resolution tile.
        self.proj2 = ConvBNAct(64, 64, 1, activation=False)
        self.proj3 = ConvBNAct(128, 64, 1, activation=False)
        self.proj4 = ConvBNAct(256, 64, 1, activation=False)
        self.high_refine = ArrayPartialSpatialBlock(64, 16)

        # One low-level detail fusion replaces V0's three serial decoder blocks.
        self.detail_proj = ConvBNAct(32, 32, 1, activation=False)
        self.semantic_proj = ConvBNAct(64, 32, 1, activation=False)
        self.detail_refine = ArrayPartialSpatialBlock(32, 16)
        self.segmentation_head = nn.Conv2d(32, num_classes, 1)

        # Training-only supervision.  It teaches the high-level decoder a mask
        # and the detail path a boundary without adding inference graph nodes.
        self.aux_coarse_head = nn.Conv2d(64, 1, 1)
        self.aux_boundary_head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _resize_like(x: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, x: Tensor, return_aux: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        input_size = x.shape[-2:]
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))

        p2 = self.proj2(x2)
        p3 = self.proj3(x3)
        p4 = self.proj4(x4)
        high = self.high_refine(p2 + self._resize_like(p3, p2) + self._resize_like(p4, p2))

        detail = self.detail_proj(x1)
        semantic = self._resize_like(self.semantic_proj(high), detail)
        decoded = self.detail_refine(detail + semantic)
        logits = F.interpolate(self.segmentation_head(decoded), size=input_size, mode="nearest")
        if not return_aux:
            return logits

        coarse = F.interpolate(self.aux_coarse_head(high), size=input_size, mode="nearest")
        boundary = F.interpolate(self.aux_boundary_head(decoded), size=input_size, mode="nearest")
        return {"segmentation": logits, "coarse": coarse, "boundary": boundary}

    @torch.no_grad()
    def switch_to_deploy(self) -> "TilePolypNetV1":
        self.eval()
        for module in self.modules():
            if isinstance(module, RepChannelMixer):
                module.switch_to_deploy()
        for module in self.modules():
            if isinstance(module, ConvBNAct) and isinstance(module[1], nn.BatchNorm2d):
                module[0] = fuse_conv_bn_eval(module[0], module[1])
                module[1] = nn.Identity()
        return self

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        auxiliary = sum(parameter.numel() for parameter in self.aux_coarse_head.parameters())
        auxiliary += sum(parameter.numel() for parameter in self.aux_boundary_head.parameters())
        return {
            "total_parameters": total,
            "active_inference_parameters": total - auxiliary,
            "training_only_aux_parameters": auxiliary,
        }

    @staticmethod
    def deployment_schedule() -> List[Dict[str, object]]:
        """Frontend-neutral fusion contract consumed by ``compile_v1.py``."""

        regions = [
            ScheduleRegion("ingress_stem", "gemmini_dense", "RGB ingress; produce NHWC16"),
            ScheduleRegion("encoder_stage1", "gemmini_pconv16", "slice+3x3+cat+1x1 fused; keep NHWC16"),
            ScheduleRegion("encoder_stage2", "gemmini_pconv16", "downsample and fused PConv16; keep tile resident"),
            ScheduleRegion("encoder_stage3", "gemmini_linear_context", "PConv32 plus KV/QKV matmuls; vector reciprocal"),
            ScheduleRegion("encoder_stage4", "gemmini_linear_context", "PConv64 plus KV/QKV matmuls; vector reciprocal"),
            ScheduleRegion("parallel_pyramid", "gemmini_stream_fuse", "parallel 1x1 projections; resize-addressed accumulation"),
            ScheduleRegion("detail_fusion", "gemmini_stream_fuse", "single semantic upsample-add and PConv16"),
            ScheduleRegion("egress_head", "gemmini_dense", "class projection then one-channel nearest egress"),
        ]
        return [asdict(region) for region in regions]


def tilepolyp_v1(**kwargs: object) -> TilePolypNetV1:
    return TilePolypNetV1(**kwargs)
