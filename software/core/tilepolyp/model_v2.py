"""ArrayPolypNet-V2: trainable hardware-native segmentation candidate.

This model is intentionally independent of the earlier TilePolypNet topology.
Its train graph contains reparameterization branches and space-to-depth views,
while ``switch_to_deploy`` lowers them to standard dense stride-2/1x1/3x3
convolutions.  The only global block is a low-resolution fixed-normalization
linear attention with one fused QKV projection and no Softmax or LayerNorm.

The module defines an algorithm candidate, not a hardware performance claim.
Physical split-accumulation, scratchpad residency, INT8 QAT and RTL cycle/Fmax
gates remain explicit downstream requirements.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .model import ConvBNAct
from .model_v1 import RepChannelMixer


ARRAY_DIM = 16


def _aligned(value: int) -> bool:
    return value % ARRAY_DIM == 0


class StaticSymmetricFakeQuant(nn.Module):
    """Explicit train-time marker for an INT8 relay/requant boundary.

    It is disabled for the initial FP32 experiment.  QAT may first enable the
    observer, freeze ``scale=max_abs/127``, then enable the straight-through
    fake quantizer.  The scale is static after calibration; no per-token runtime
    quantization is implied.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("observed_max", torch.tensor(0.0, dtype=torch.float32))
        self.observer_enabled = False
        self.fake_quant_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        if self.observer_enabled:
            current = x.detach().abs().amax().to(self.observed_max.dtype)
            self.observed_max.copy_(torch.maximum(self.observed_max, current))
        if not self.fake_quant_enabled:
            return x
        scale = self.scale.clamp_min(torch.finfo(x.dtype).eps).to(dtype=x.dtype)
        quantized = torch.clamp(torch.round(x / scale), -127, 127)
        dequantized = quantized * scale
        return x + (dequantized - x).detach()

    @torch.no_grad()
    def reset_observer(self) -> None:
        self.observed_max.zero_()

    @torch.no_grad()
    def freeze_and_enable(self) -> None:
        self.scale.copy_((self.observed_max / 127.0).clamp_min(1e-8))
        self.observer_enabled = False
        self.fake_quant_enabled = True


class SpaceToDepthDownsample(nn.Module):
    """Train as pixel-unshuffle + 1x1, deploy as one dense 2x2/stride-2 Conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if not _aligned(out_channels):
            raise ValueError("out_channels must be a multiple of 16")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj: Optional[nn.Conv2d] = nn.Conv2d(
            in_channels * 4, out_channels, kernel_size=1, bias=False
        )
        self.bn: Optional[nn.BatchNorm2d] = nn.BatchNorm2d(out_channels)
        self.deploy_conv: Optional[nn.Conv2d] = None
        self.activation = nn.ReLU6(inplace=False)

    @property
    def is_deploy(self) -> bool:
        return self.deploy_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy_conv is not None:
            return self.activation(self.deploy_conv(x))
        assert self.proj is not None and self.bn is not None
        return self.activation(self.bn(self.proj(F.pixel_unshuffle(x, 2))))

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy_conv is not None:
            return
        assert self.proj is not None and self.bn is not None
        fused = fuse_conv_bn_eval(self.proj, self.bn)
        deploy = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        ).to(device=fused.weight.device, dtype=fused.weight.dtype)
        deploy.weight.copy_(
            fused.weight[:, :, 0, 0].reshape(
                self.out_channels, self.in_channels, 2, 2
            )
        )
        assert fused.bias is not None and deploy.bias is not None
        deploy.bias.copy_(fused.bias)
        self.deploy_conv = deploy
        self.proj = None
        self.bn = None


class RepSpatialConv(nn.Module):
    """3x3 + 1x1 + identity BN branches that deploy as one dense 3x3."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if not _aligned(channels):
            raise ValueError("channels must be a multiple of 16")
        self.channels = channels
        self.branch3: Optional[nn.Sequential] = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.branch1: Optional[nn.Sequential] = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.identity_bn: Optional[nn.BatchNorm2d] = nn.BatchNorm2d(channels)
        self.deploy_conv: Optional[nn.Conv2d] = None

    @property
    def is_deploy(self) -> bool:
        return self.deploy_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        assert self.branch3 is not None
        assert self.branch1 is not None
        assert self.identity_bn is not None
        return self.branch3(x) + self.branch1(x) + self.identity_bn(x)

    @staticmethod
    def _identity_kernel_bias(bn: nn.BatchNorm2d) -> Tuple[Tensor, Tensor]:
        kernel = torch.zeros(
            bn.num_features,
            bn.num_features,
            3,
            3,
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        index = torch.arange(bn.num_features, device=kernel.device)
        kernel[index, index, 1, 1] = 1.0
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        kernel = kernel * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy_conv is not None:
            return
        assert self.branch3 is not None
        assert self.branch1 is not None
        assert self.identity_bn is not None
        conv3 = fuse_conv_bn_eval(self.branch3[0], self.branch3[1])
        conv1 = fuse_conv_bn_eval(self.branch1[0], self.branch1[1])
        identity_kernel, identity_bias = self._identity_kernel_bias(self.identity_bn)
        deploy = nn.Conv2d(
            self.channels, self.channels, 3, padding=1, bias=True
        ).to(device=conv3.weight.device, dtype=conv3.weight.dtype)
        kernel1 = F.pad(conv1.weight, (1, 1, 1, 1))
        assert conv3.bias is not None and conv1.bias is not None and deploy.bias is not None
        deploy.weight.copy_(conv3.weight + kernel1 + identity_kernel)
        deploy.bias.copy_(conv3.bias + conv1.bias + identity_bias)
        self.deploy_conv = deploy
        self.branch3 = None
        self.branch1 = None
        self.identity_bn = None


class ArrayRepBlock(nn.Module):
    """Reparameterized spatial mixing on one or two 16-channel array tiles.

    The logical ``cat`` is a compiler contract: the following 1x1 consumes the
    processed and bypass slices through split accumulation into one accumulator.
    It must not become a physical packed tensor in the hardware deploy graph.
    """

    def __init__(self, channels: int, spatial_channels: int) -> None:
        super().__init__()
        if not _aligned(channels) or not _aligned(spatial_channels):
            raise ValueError("channels and spatial_channels must be 16-aligned")
        if not 0 < spatial_channels <= channels:
            raise ValueError("spatial_channels must be in (0, channels]")
        self.channels = channels
        self.spatial_channels = spatial_channels
        self.spatial = RepSpatialConv(spatial_channels)
        self.spatial_activation = nn.ReLU6(inplace=False)
        self.channel_mixer = RepChannelMixer(channels)
        self.output_activation = nn.ReLU6(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial_activation(self.spatial(x[:, : self.spatial_channels]))
        if self.spatial_channels == self.channels:
            mixed = spatial
        else:
            mixed = torch.cat((spatial, x[:, self.spatial_channels :]), dim=1)
        return self.output_activation(self.channel_mixer(mixed))

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.spatial.switch_to_deploy()
        self.channel_mixer.switch_to_deploy()


class RepFullBlock(nn.Module):
    """One full 16-channel train-time RepConv, one 3x3 at deployment."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        if not _aligned(channels):
            raise ValueError("channels must be 16-aligned")
        self.spatial = RepSpatialConv(channels)
        self.activation = nn.ReLU6(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.spatial(x))

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.spatial.switch_to_deploy()


class FixedLinearAttention(nn.Module):
    """Low-resolution linear attention with fused QKV and fixed normalization.

    Q/K/V are generated in one 1x1 convolution.  Q and K use a non-negative
    ReLU6 feature map, and the N-dependent normalization is a compile-time
    constant that can be folded into the output projection.  ``kv_requant`` is
    the explicit QAT boundary required before the second array MatMul.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int = 32,
        heads: int = 2,
        expected_tokens: int = 256,
    ) -> None:
        super().__init__()
        if not _aligned(channels) or not _aligned(attention_channels):
            raise ValueError("attention dimensions must be 16-aligned")
        if attention_channels % heads:
            raise ValueError("attention_channels must be divisible by heads")
        head_dim = attention_channels // heads
        if not _aligned(head_dim):
            raise ValueError("head_dim must be a multiple of 16")
        self.channels = channels
        self.attention_channels = attention_channels
        self.heads = heads
        self.head_dim = head_dim
        self.kv_channels = head_dim
        self.expected_tokens = expected_tokens
        self.qkv_conv: nn.Module = nn.Conv2d(
            channels, attention_channels + 2 * head_dim, 1, bias=False
        )
        self.qkv_bn: nn.Module = nn.BatchNorm2d(attention_channels + 2 * head_dim)
        self.kv_requant = StaticSymmetricFakeQuant()
        self.head_scale: Optional[nn.Parameter] = nn.Parameter(
            torch.full((heads,), head_dim**-0.5, dtype=torch.float32)
        )
        self.out_conv: nn.Module = nn.Conv2d(
            attention_channels, channels, 1, bias=False
        )
        self.out_bn: nn.Module = nn.BatchNorm2d(channels)
        self.output_activation = nn.ReLU6(inplace=False)
        self.scale_folded = False

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        tokens = height * width
        if tokens != self.expected_tokens:
            raise ValueError(
                f"attention expected {self.expected_tokens} tokens, received {tokens}"
            )
        qkv = self.qkv_bn(self.qkv_conv(x))
        q, k, v = torch.split(
            qkv,
            (self.attention_channels, self.kv_channels, self.kv_channels),
            dim=1,
        )

        def query_view(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch, self.heads, self.head_dim, tokens
            ).transpose(-2, -1)

        def shared_kv_view(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, 1, self.head_dim, tokens).transpose(-2, -1)

        q = query_view(torch.clamp(q, 0.0, 6.0))
        k = shared_kv_view(torch.clamp(k, 0.0, 6.0))
        v = shared_kv_view(torch.clamp(v, -6.0, 6.0))
        # Apply the fixed normalization before the second MatMul while training.
        # This is algebraically identical to scaling its output, but keeps the
        # FP16 Q(K^T V) accumulator away from overflow under autocast.  The
        # deploy graph still folds the same constant into ``out_conv``.
        if not self.scale_folded:
            assert self.head_scale is not None
            scale = self.head_scale.reshape(1, self.heads, 1, 1) / float(tokens)
            q = q * scale
        kv = torch.matmul(k.transpose(-2, -1), v)
        kv = self.kv_requant(kv)
        context = torch.matmul(q, kv)
        context = context.transpose(-2, -1).reshape(
            batch, self.attention_channels, height, width
        )
        projected = self.out_bn(self.out_conv(context))
        return self.output_activation(x + projected)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.scale_folded:
            return
        assert isinstance(self.qkv_conv, nn.Conv2d)
        assert isinstance(self.qkv_bn, nn.BatchNorm2d)
        assert isinstance(self.out_conv, nn.Conv2d)
        assert isinstance(self.out_bn, nn.BatchNorm2d)
        assert self.head_scale is not None
        self.qkv_conv = fuse_conv_bn_eval(self.qkv_conv, self.qkv_bn)
        self.qkv_bn = nn.Identity()
        fused_out = fuse_conv_bn_eval(self.out_conv, self.out_bn)
        input_scale = (
            self.head_scale.detach().repeat_interleave(self.head_dim)
            / float(self.expected_tokens)
        )
        fused_out.weight.mul_(input_scale.reshape(1, -1, 1, 1))
        self.out_conv = fused_out
        self.out_bn = nn.Identity()
        self.register_parameter("head_scale", None)
        self.scale_folded = True

    def matmul_macs(self, batch: int = 1) -> int:
        # One shared K^T V plus one Q(K^T V) per query head.
        return (
            batch
            * (1 + self.heads)
            * self.expected_tokens
            * self.head_dim**2
        )


@dataclass(frozen=True)
class V2Region:
    name: str
    backend: str
    contract: str
    persistent_reads: Tuple[str, ...] = ()
    persistent_writes: Tuple[str, ...] = ()


class ArrayPolypNetV2(nn.Module):
    """First trainable candidate derived from the 300-FPS co-design gates."""

    channel_plan: Tuple[int, ...] = (16, 32, 64, 96, 128)
    skip_channel_plan: Tuple[int, ...] = (16, 16, 16, 16)

    def __init__(
        self,
        num_classes: int = 1,
        input_size: int = 256,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if input_size % 32:
            raise ValueError("input_size must be divisible by 32")
        self.num_classes = num_classes
        self.input_size = input_size
        self.use_attention = use_attention

        # Keep one overlapping RGB stem for boundary fidelity.  Every later
        # train-time reshape disappears into a dense 2x2/stride-2 kernel.
        self.down1 = ConvBNAct(3, 16, 3, stride=2)   # 256 -> 128
        self.down2 = SpaceToDepthDownsample(16, 32)  # 128 -> 64
        self.stage1 = nn.Sequential(ArrayRepBlock(32, 16))
        self.skip1_encode = ConvBNAct(32, 16, 1, activation=False)

        self.down3 = SpaceToDepthDownsample(32, 64)  # 64 -> 32
        self.stage2 = nn.Sequential(
            ArrayRepBlock(64, 16),
            ArrayRepBlock(64, 16),
        )
        self.skip2_encode = ConvBNAct(64, 16, 1, activation=False)

        self.down4 = SpaceToDepthDownsample(64, 96)  # 32 -> 16
        stage3: List[nn.Module] = [
            ArrayRepBlock(96, 16),
            ArrayRepBlock(96, 16),
        ]
        if use_attention:
            stage3.append(
                FixedLinearAttention(
                    96,
                    attention_channels=32,
                    heads=2,
                    expected_tokens=(input_size // 16) ** 2,
                )
            )
        self.stage3 = nn.Sequential(*stage3)
        self.skip3_encode = ConvBNAct(96, 16, 1, activation=False)

        self.down5 = SpaceToDepthDownsample(96, 128)  # 16 -> 8
        self.stage4 = nn.Sequential(ArrayRepBlock(128, 32))
        self.skip4_encode = ConvBNAct(128, 16, 1, activation=False)

        self.high_refine = RepFullBlock(16)
        self.semantic_encode = ConvBNAct(16, 16, 1, activation=False)
        self.detail_refine = RepFullBlock(16)
        self.segmentation_head = nn.Conv2d(16, num_classes, 1)

        # Training-only supervision; these heads are not called by deploy forward.
        self.aux_coarse_head = nn.Conv2d(16, 1, 1)
        self.aux_boundary_head = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _resize_like(x: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size}, got {tuple(x.shape[-2:])}"
            )
        input_size = x.shape[-2:]
        x1 = self.stage1(self.down2(self.down1(x)))
        skip1 = self.skip1_encode(x1)
        x2 = self.stage2(self.down3(x1))
        skip2 = self.skip2_encode(x2)
        x3 = self.stage3(self.down4(x2))
        skip3 = self.skip3_encode(x3)
        x4 = self.stage4(self.down5(x3))
        skip4 = self.skip4_encode(x4)

        high = skip2 + self._resize_like(skip3, skip2) + self._resize_like(skip4, skip2)
        high = self.high_refine(high)
        semantic = self._resize_like(self.semantic_encode(high), skip1)
        detail = self.detail_refine(skip1 + semantic)
        logits = F.interpolate(
            self.segmentation_head(detail), size=input_size, mode="nearest"
        )
        if not return_aux:
            return logits
        coarse = F.interpolate(
            self.aux_coarse_head(high), size=input_size, mode="nearest"
        )
        boundary = F.interpolate(
            self.aux_boundary_head(detail), size=input_size, mode="nearest"
        )
        return {"segmentation": logits, "coarse": coarse, "boundary": boundary}

    @torch.no_grad()
    def switch_to_deploy(self) -> "ArrayPolypNetV2":
        self.eval()
        modules = list(self.modules())
        for module in modules:
            if isinstance(module, SpaceToDepthDownsample):
                module.switch_to_deploy()
            elif isinstance(module, ArrayRepBlock):
                module.switch_to_deploy()
            elif isinstance(module, RepFullBlock):
                module.switch_to_deploy()
            elif isinstance(module, FixedLinearAttention):
                module.switch_to_deploy()
        for module in list(self.modules()):
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
    def skip_capsules(input_size: int = 256) -> List[Dict[str, object]]:
        shapes = (
            ("skip1", (1, 16, input_size // 4, input_size // 4), 2, 0),
            ("skip2", (1, 16, input_size // 8, input_size // 8), 3, 0),
            ("skip3", (1, 16, input_size // 16, input_size // 16), 3, 16 * 1024),
            ("skip4", (1, 16, input_size // 32, input_size // 32), 3, 20 * 1024),
        )
        result: List[Dict[str, object]] = []
        for name, shape, bank, offset in shapes:
            size = math.prod(shape)
            result.append(
                {
                    "name": name,
                    "shape": shape,
                    "bytes": size,
                    "bank": bank,
                    "bank_offset_bytes": offset,
                }
            )
        return result

    @staticmethod
    def deployment_schedule(use_attention: bool = True) -> List[Dict[str, object]]:
        regions = [
            V2Region(
                "reshape_stem_stage1",
                "dense_spad_pipeline",
                "overlapping 3x3/stride-2 RGB stem; implicit 2x2 downsample; RepSpatial16",
                persistent_writes=("skip1",),
            ),
            V2Region(
                "stage2_fork",
                "dense_spad_pipeline",
                "implicit space-to-depth downsample; two RepSpatial16 regions",
                persistent_writes=("skip2",),
            ),
            V2Region(
                "stage3_context_fork",
                "dense_spad_pipeline+linear_attention" if use_attention else "dense_spad_pipeline",
                "fused QKV; descriptor heads; K^T V INT8 relay; no Softmax/LN",
                persistent_writes=("skip3",),
            ),
            V2Region(
                "stage4_fork",
                "dense_spad_pipeline",
                "8x8 semantic block and compressed capsule",
                persistent_writes=("skip4",),
            ),
            V2Region(
                "pyramid_consume",
                "resize_addressed_split_accumulate",
                "three scale-aligned sources; no physical resize/concat",
                persistent_reads=("skip2", "skip3", "skip4"),
            ),
            V2Region(
                "detail_consume",
                "resize_addressed_split_accumulate",
                "project at 32x32, broadcast to 64x64, consume skip1 and release",
                persistent_reads=("skip1",),
            ),
            V2Region(
                "class_egress",
                "vector_dot+scatter",
                "16-to-class projection at 64x64; repeated 4x4 output scatter",
            ),
        ]
        return [asdict(region) for region in regions]


def arraypolyp_v2(**kwargs: object) -> ArrayPolypNetV2:
    return ArrayPolypNetV2(**kwargs)
