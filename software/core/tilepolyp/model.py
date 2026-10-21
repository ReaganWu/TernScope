"""TilePolypNet-V0: a profiling-first hardware/software co-design prototype.

This is deliberately an untrained research prototype.  Its operator grammar is
chosen to expose a clean heterogeneous hardware mapping:

* dense 1x1 convolutions and attention matmuls -> systolic array;
* depthwise spatial filters and resize -> streaming spatial engine;
* softmax/elementwise work -> vector/reduction engine;
* training-only linear factorization -> exactly folded deployment weights.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.profiler import record_function


def _activation(enabled: bool = True) -> nn.Module:
    # ReLU6 is supported by Gemmini's standard peripheral activation path.
    return nn.ReLU6(inplace=False) if enabled else nn.Identity()


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            _activation(activation),
        )


class DepthwiseSeparableBlock(nn.Module):
    """A line-buffer-friendly spatial filter followed by dense channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1, activation=False)
        self.out_act = _activation(True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pointwise(self.depthwise(x))
        if self.use_residual:
            y = y + x
        return self.out_act(y)


class RepPointwise(nn.Module):
    """Two linear 1x1 convolutions that fold exactly into one at deployment.

    There is intentionally no normalization or activation between ``expand``
    and ``project``.  The training parameterization is therefore exactly
    representable by a single affine 1x1 convolution during inference.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.expand: Optional[nn.Conv2d] = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)
        self.project: Optional[nn.Conv2d] = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)
        self.deploy: Optional[nn.Conv2d] = None

    @property
    def is_deploy(self) -> bool:
        return self.deploy is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy is not None:
            return self.deploy(x)
        assert self.expand is not None and self.project is not None
        return self.project(self.expand(x))

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy is not None:
            return
        assert self.expand is not None and self.project is not None
        w1 = self.expand.weight[:, :, 0, 0]
        w2 = self.project.weight[:, :, 0, 0]
        b1 = self.expand.bias
        b2 = self.project.bias
        assert b1 is not None and b2 is not None

        fused = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=True)
        fused = fused.to(device=w1.device, dtype=w1.dtype)
        fused.weight.copy_((w2 @ w1).view(self.out_channels, self.in_channels, 1, 1))
        fused.bias.copy_(b2 + w2 @ b1)
        self.deploy = fused
        self.expand = None
        self.project = None


class TileContextBlock(nn.Module):
    """Local-global block with a fixed, array-aligned context-token count.

    For the default 8x8 context grid, K/V contain 64 tokens.  Channel widths and
    per-head dimensions are chosen as multiples of 16 in ``TilePolypNetV0``.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        context_grid: int = 8,
        normalizer: str = "softmax",
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        if (channels // heads) % 16 != 0:
            raise ValueError("per-head dimension must be a multiple of 16")
        if normalizer not in {"softmax", "relu_l1"}:
            raise ValueError("normalizer must be 'softmax' or 'relu_l1'")

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.context_grid = context_grid
        self.context_tokens = context_grid * context_grid
        self.normalizer = normalizer

        self.local = DepthwiseSeparableBlock(channels, channels)
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.kv_proj = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.out_proj = ConvBNAct(channels, channels, 1, activation=False)
        self.out_act = _activation(True)

    def _normalize(self, scores: Tensor) -> Tensor:
        if self.normalizer == "softmax":
            return scores.softmax(dim=-1)
        weights = F.relu(scores)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width

        with record_function("tile_context.local_stream"):
            local = self.local(x)
        with record_function("tile_context.context_pool"):
            context = F.adaptive_avg_pool2d(x, (self.context_grid, self.context_grid))
        with record_function("tile_context.qkv_projection"):
            q = self.q_proj(x)
            kv = self.kv_proj(context)
            k, v = kv.chunk(2, dim=1)
            q = q.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
            k = k.reshape(batch, self.heads, self.head_dim, self.context_tokens).transpose(-2, -1)
            v = v.reshape(batch, self.heads, self.head_dim, self.context_tokens).transpose(-2, -1)
        with record_function("tile_context.qk_matmul"):
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        with record_function("tile_context.normalize"):
            attention = self._normalize(scores)
        with record_function("tile_context.av_matmul"):
            global_features = torch.matmul(attention, v)
        with record_function("tile_context.output_projection"):
            global_features = global_features.transpose(-2, -1).reshape(batch, channels, height, width)
            global_features = self.out_proj(global_features)
        return self.out_act(x + local + global_features)

    def attention_macs(self, batch: int, height: int, width: int) -> int:
        # QK^T and AV each cost B * heads * N * T * head_dim MACs.
        return 2 * batch * self.heads * height * width * self.context_tokens * self.head_dim


class TilePolypNetV0(nn.Module):
    """A compact hardware-native binary/multiclass segmentation prototype."""

    channel_plan: Tuple[int, ...] = (16, 32, 64, 128, 256)

    def __init__(
        self,
        num_classes: int = 1,
        decoder_channels: int = 64,
        context_grid: int = 8,
        normalizer: str = "softmax",
        use_global_context: bool = True,
    ) -> None:
        super().__init__()
        if decoder_channels % 16 != 0:
            raise ValueError("decoder_channels must be a multiple of 16")
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        self.use_global_context = use_global_context

        self.stem = nn.Sequential(
            ConvBNAct(3, 16, 3, stride=2),
            DepthwiseSeparableBlock(16, 32, stride=2),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(32, 32),
            DepthwiseSeparableBlock(32, 32),
        )

        self.down2 = DepthwiseSeparableBlock(32, 64, stride=2)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 64),
            TileContextBlock(64, heads=4, context_grid=context_grid, normalizer=normalizer)
            if use_global_context
            else DepthwiseSeparableBlock(64, 64),
        )

        self.down3 = DepthwiseSeparableBlock(64, 128, stride=2)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(128, 128),
            TileContextBlock(128, heads=8, context_grid=context_grid, normalizer=normalizer)
            if use_global_context
            else DepthwiseSeparableBlock(128, 128),
        )

        self.down4 = DepthwiseSeparableBlock(128, 256, stride=2)
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(256, 256),
            TileContextBlock(256, heads=8, context_grid=context_grid, normalizer=normalizer)
            if use_global_context
            else DepthwiseSeparableBlock(256, 256),
        )

        # Linear training-time factorizations; all four collapse to one 1x1 conv.
        self.proj1 = RepPointwise(32, decoder_channels, hidden_channels=64)
        self.proj2 = RepPointwise(64, decoder_channels, hidden_channels=64)
        self.proj3 = RepPointwise(128, decoder_channels, hidden_channels=128)
        self.proj4 = RepPointwise(256, decoder_channels, hidden_channels=256)

        self.fuse3 = DepthwiseSeparableBlock(decoder_channels, decoder_channels)
        self.fuse2 = DepthwiseSeparableBlock(decoder_channels, decoder_channels)
        self.fuse1 = DepthwiseSeparableBlock(decoder_channels, decoder_channels)
        self.segmentation_head = nn.Conv2d(decoder_channels, num_classes, 1)

        # An explicitly training-only pseudo-depth supervision head.
        self.aux_depth_head = nn.Conv2d(decoder_channels, 1, 1)

    @staticmethod
    def _resize_like(x: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    @staticmethod
    def _run_timed(
        name: str,
        fn: Callable[[], Tensor],
        timings: Optional[DefaultDict[str, List[float]]],
        synchronize: Optional[Callable[[], None]],
    ) -> Tensor:
        if timings is None:
            with record_function(name):
                return fn()
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        with record_function(name):
            output = fn()
        if synchronize is not None:
            synchronize()
        timings[name].append((time.perf_counter() - start) * 1000.0)
        return output

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
        stage_timings: Optional[DefaultDict[str, List[float]]] = None,
        synchronize: Optional[Callable[[], None]] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        input_size = x.shape[-2:]
        run = lambda name, fn: self._run_timed(name, fn, stage_timings, synchronize)

        x1 = run("model.stem_stage1", lambda: self.stage1(self.stem(x)))
        x2 = run("model.stage2", lambda: self.stage2(self.down2(x1)))
        x3 = run("model.stage3", lambda: self.stage3(self.down3(x2)))
        x4 = run("model.stage4", lambda: self.stage4(self.down4(x3)))

        p1 = run("decoder.proj1", lambda: self.proj1(x1))
        p2 = run("decoder.proj2", lambda: self.proj2(x2))
        p3 = run("decoder.proj3", lambda: self.proj3(x3))
        p4 = run("decoder.proj4", lambda: self.proj4(x4))

        d3 = run("decoder.fuse3", lambda: self.fuse3(self._resize_like(p4, p3) + p3))
        d2 = run("decoder.fuse2", lambda: self.fuse2(self._resize_like(d3, p2) + p2))
        d1 = run("decoder.fuse1", lambda: self.fuse1(self._resize_like(d2, p1) + p1))
        logits = run(
            "decoder.head",
            # Project 64 channels to the class logits before resizing.  Reversing
            # this order would materialize a full-resolution 64-channel tensor.
            lambda: F.interpolate(
                self.segmentation_head(d1), size=input_size, mode="nearest"
            ),
        )

        if not return_aux:
            return logits
        depth = run(
            "training_only.depth_head",
            lambda: F.interpolate(self.aux_depth_head(d3), size=input_size, mode="nearest"),
        )
        return {"segmentation": logits, "pseudo_depth": depth}

    @torch.no_grad()
    def switch_to_deploy(self) -> "TilePolypNetV0":
        self.eval()
        for module in self.modules():
            if isinstance(module, RepPointwise):
                module.switch_to_deploy()
        # BatchNorm is affine and constant in eval mode; fold it into the
        # preceding convolution to remove standalone normalization operators.
        for module in self.modules():
            if isinstance(module, ConvBNAct) and isinstance(module[1], nn.BatchNorm2d):
                module[0] = fuse_conv_bn_eval(module[0], module[1])
                module[1] = nn.Identity()
        return self

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        auxiliary = sum(parameter.numel() for parameter in self.aux_depth_head.parameters())
        return {
            "total_parameters": total,
            "active_inference_parameters": total - auxiliary,
            "training_only_aux_parameters": auxiliary,
        }


def tilepolyp_v0(**kwargs: object) -> TilePolypNetV0:
    return TilePolypNetV0(**kwargs)
