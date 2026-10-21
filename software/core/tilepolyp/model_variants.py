"""Gemmini-oriented ArrayPolypNet exploration variants.

The variants intentionally share one decoder and one Kvasir training protocol,
but exercise different encoder operator grammars.  Every internal channel count
is aligned to a 16x16 systolic array.  No variant uses grouped/depthwise
convolution, Softmax, LayerNorm, or dynamic normalization.
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
from .model_v2 import (
    ARRAY_DIM,
    ArrayRepBlock,
    FixedLinearAttention,
    RepFullBlock,
    RepSpatialConv,
    SpaceToDepthDownsample,
    StaticSymmetricFakeQuant,
)


def _aligned(value: int) -> bool:
    return value % ARRAY_DIM == 0


def _pad_kernel(kernel: Tensor, target_size: int) -> Tensor:
    difference = target_size - kernel.shape[-1]
    if difference < 0 or difference % 2:
        raise ValueError("kernel cannot be centered in target kernel")
    padding = difference // 2
    return F.pad(kernel, (padding, padding, padding, padding))


class RepLargeKernelConv(nn.Module):
    """Multi-branch large kernel that becomes one dense KxK convolution.

    KxK, (K-2)x(K-2), 3x3, 1x1 and identity-BN branches are used when
    distinct.  All branches are linear and therefore exactly mergeable.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if not _aligned(channels):
            raise ValueError("channels must be a multiple of 16")
        if kernel_size < 5 or kernel_size % 2 == 0:
            raise ValueError("large kernel must be odd and at least 5")
        self.channels = channels
        self.kernel_size = kernel_size
        branch_sizes: List[int] = []
        for size in (kernel_size, kernel_size - 2, 3, 1):
            if size not in branch_sizes:
                branch_sizes.append(size)
        self.branches: Optional[nn.ModuleList] = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        size,
                        padding=size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                )
                for size in branch_sizes
            ]
        )
        self.identity_bn: Optional[nn.BatchNorm2d] = nn.BatchNorm2d(channels)
        self.deploy_conv: Optional[nn.Conv2d] = None

    @property
    def is_deploy(self) -> bool:
        return self.deploy_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        assert self.branches is not None and self.identity_bn is not None
        result = self.identity_bn(x)
        for branch in self.branches:
            result = result + branch(x)
        return result

    def _identity_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        assert self.identity_bn is not None
        bn = self.identity_bn
        kernel = torch.zeros(
            self.channels,
            self.channels,
            self.kernel_size,
            self.kernel_size,
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        index = torch.arange(self.channels, device=kernel.device)
        center = self.kernel_size // 2
        kernel[index, index, center, center] = 1.0
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        kernel.mul_(scale.reshape(-1, 1, 1, 1))
        bias = bn.bias - bn.running_mean * scale
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy_conv is not None:
            return
        assert self.branches is not None and self.identity_bn is not None
        kernel, bias = self._identity_kernel_bias()
        for branch in self.branches:
            fused = fuse_conv_bn_eval(branch[0], branch[1])
            kernel = kernel + _pad_kernel(fused.weight, self.kernel_size)
            assert fused.bias is not None
            bias = bias + fused.bias
        deploy = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        deploy.weight.copy_(kernel)
        assert deploy.bias is not None
        deploy.bias.copy_(bias)
        self.deploy_conv = deploy
        self.branches = None
        self.identity_bn = None


class RepStride4Stem(nn.Module):
    """5x5/3x3/1x1 RGB branches that become one 5x5/stride-4 conv.

    No consumer observes the original shell's intermediate 128x128 tensor.
    The Gemmini-specific graph therefore downsamples directly to 64x64 while
    retaining overlapping 5x5 support and train-time multi-scale branches.
    """

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        if not _aligned(out_channels):
            raise ValueError("stem channels must be a multiple of 16")
        self.out_channels = out_channels
        self.branches: Optional[nn.ModuleList] = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3,
                        out_channels,
                        kernel_size=size,
                        stride=4,
                        padding=size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
                for size in (5, 3, 1)
            ]
        )
        self.deploy_conv: Optional[nn.Conv2d] = None
        self.activation = nn.ReLU6(inplace=False)

    @property
    def is_deploy(self) -> bool:
        return self.deploy_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy_conv is not None:
            return self.activation(self.deploy_conv(x))
        assert self.branches is not None
        result = self.branches[0](x)
        for branch in self.branches[1:]:
            result = result + branch(x)
        return self.activation(result)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy_conv is not None:
            return
        assert self.branches is not None
        fused = [fuse_conv_bn_eval(branch[0], branch[1]) for branch in self.branches]
        kernel = fused[0].weight.clone()
        assert fused[0].bias is not None
        bias = fused[0].bias.clone()
        for branch in fused[1:]:
            kernel.add_(_pad_kernel(branch.weight, 5))
            assert branch.bias is not None
            bias.add_(branch.bias)
        deploy = nn.Conv2d(
            3,
            self.out_channels,
            kernel_size=5,
            stride=4,
            padding=2,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        deploy.weight.copy_(kernel)
        assert deploy.bias is not None
        deploy.bias.copy_(bias)
        self.deploy_conv = deploy
        self.branches = None


@torch.no_grad()
def _compose_pointwise_before_conv(
    pointwise: nn.Conv2d, consumer: nn.Conv2d
) -> nn.Conv2d:
    """Fold an affine 1x1 producer into an unpadded convolution consumer."""

    if pointwise.kernel_size != (1, 1) or pointwise.groups != 1:
        raise ValueError("producer must be one dense 1x1 convolution")
    if consumer.groups != 1 or consumer.padding != (0, 0):
        raise ValueError("consumer fold requires one dense unpadded convolution")
    if pointwise.out_channels != consumer.in_channels:
        raise ValueError("producer/consumer channel mismatch")
    mixer_weight = pointwise.weight[:, :, 0, 0]
    kernel = torch.einsum("omhw,mi->oihw", consumer.weight, mixer_weight)
    mixer_bias = (
        pointwise.bias
        if pointwise.bias is not None
        else torch.zeros(
            pointwise.out_channels,
            device=kernel.device,
            dtype=kernel.dtype,
        )
    )
    consumer_bias = (
        consumer.bias
        if consumer.bias is not None
        else torch.zeros(
            consumer.out_channels,
            device=kernel.device,
            dtype=kernel.dtype,
        )
    )
    bias = consumer_bias + consumer.weight.sum(dim=(2, 3)) @ mixer_bias
    folded = nn.Conv2d(
        pointwise.in_channels,
        consumer.out_channels,
        kernel_size=consumer.kernel_size,
        stride=consumer.stride,
        padding=consumer.padding,
        dilation=consumer.dilation,
        groups=1,
        bias=True,
    ).to(device=kernel.device, dtype=kernel.dtype)
    folded.weight.copy_(kernel)
    assert folded.bias is not None
    folded.bias.copy_(bias)
    return folded


class RepLKConsumerFoldBlock(nn.Module):
    """Partial RepLK followed by a linear mixer folded into both consumers."""

    def __init__(self, channels: int, spatial_channels: int, kernel_size: int) -> None:
        super().__init__()
        if not _aligned(channels) or not _aligned(spatial_channels):
            raise ValueError("channels must be array aligned")
        if not 0 < spatial_channels <= channels:
            raise ValueError("invalid spatial channel count")
        self.channels = channels
        self.spatial_channels = spatial_channels
        self.spatial = RepLargeKernelConv(spatial_channels, kernel_size)
        self.spatial_activation = nn.ReLU6(inplace=False)
        self.channel_mixer: nn.Module = RepChannelMixer(channels)
        self.emit_pre_mixer = False

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial_activation(self.spatial(x[:, : self.spatial_channels]))
        mixed = (
            spatial
            if self.spatial_channels == self.channels
            else torch.cat((spatial, x[:, self.spatial_channels :]), dim=1)
        )
        return mixed if self.emit_pre_mixer else self.channel_mixer(mixed)

    @torch.no_grad()
    def take_deploy_mixer(self) -> nn.Conv2d:
        if self.emit_pre_mixer:
            raise RuntimeError("deploy mixer was already consumed")
        if not isinstance(self.channel_mixer, RepChannelMixer):
            raise TypeError("unexpected channel mixer")
        self.channel_mixer.switch_to_deploy()
        if not isinstance(self.channel_mixer.conv, nn.Conv2d):
            raise TypeError("channel mixer did not lower to Conv2d")
        mixer = self.channel_mixer.conv
        self.channel_mixer = nn.Identity()
        self.emit_pre_mixer = True
        return mixer


class RepLKTileBlock(nn.Module):
    """Large-kernel mixing on one or two array tiles plus folded PW residual."""

    def __init__(self, channels: int, spatial_channels: int, kernel_size: int) -> None:
        super().__init__()
        if not _aligned(channels) or not _aligned(spatial_channels):
            raise ValueError("channels must be array aligned")
        if not 0 < spatial_channels <= channels:
            raise ValueError("invalid spatial channel count")
        self.channels = channels
        self.spatial_channels = spatial_channels
        self.spatial = RepLargeKernelConv(spatial_channels, kernel_size)
        self.spatial_activation = nn.ReLU6(inplace=False)
        self.channel_mixer = RepChannelMixer(channels)
        self.output_activation = nn.ReLU6(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial_activation(self.spatial(x[:, : self.spatial_channels]))
        mixed = (
            spatial
            if self.spatial_channels == self.channels
            else torch.cat((spatial, x[:, self.spatial_channels :]), dim=1)
        )
        return self.output_activation(self.channel_mixer(mixed))


class StarPWBlock(nn.Module):
    """Rep-spatial front end plus one fused StarNet-style pointwise gate."""

    def __init__(self, channels: int, spatial_channels: int, gate_channels: int) -> None:
        super().__init__()
        if not all(_aligned(value) for value in (channels, spatial_channels, gate_channels)):
            raise ValueError("Star block dimensions must be 16 aligned")
        self.channels = channels
        self.spatial_channels = spatial_channels
        self.gate_channels = gate_channels
        self.spatial = RepSpatialConv(spatial_channels)
        self.spatial_activation = nn.ReLU6(inplace=False)
        self.gate = ConvBNAct(channels, 2 * gate_channels, 1, activation=False)
        self.product_requant = StaticSymmetricFakeQuant()
        self.project = ConvBNAct(gate_channels, channels, 1, activation=False)
        self.output_activation = nn.ReLU6(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial_activation(self.spatial(x[:, : self.spatial_channels]))
        mixed = (
            spatial
            if self.spatial_channels == self.channels
            else torch.cat((spatial, x[:, self.spatial_channels :]), dim=1)
        )
        left, right = self.gate(mixed).chunk(2, dim=1)
        star = F.relu6(left) * torch.clamp(right, -6.0, 6.0)
        star = self.product_requant(star)
        return self.output_activation(x + self.project(star))


class PointwiseMatrixMLP(nn.Module):
    """Pure dense 1x1 residual MLP; a high-utilization GEMM control."""

    def __init__(self, channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        if not _aligned(channels) or not _aligned(bottleneck_channels):
            raise ValueError("MLP dimensions must be 16 aligned")
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.reduce = ConvBNAct(channels, bottleneck_channels, 1)
        self.expand = ConvBNAct(bottleneck_channels, channels, 1, activation=False)
        self.output_activation = nn.ReLU6(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(x + self.expand(self.reduce(x)))


class CrossScaleLinearAttention(nn.Module):
    """16x16 queries attend to an 8x8 context without Softmax or LayerNorm.

    This is one aligned d=32 linear-attention stream, not multi-query attention;
    the name is kept precise so later head-count experiments are not conflated.
    """

    def __init__(
        self,
        query_channels: int = 96,
        context_channels: int = 128,
        attention_channels: int = 32,
        expected_query_tokens: int = 256,
        expected_context_tokens: int = 64,
    ) -> None:
        super().__init__()
        if not all(
            _aligned(value)
            for value in (query_channels, context_channels, attention_channels)
        ):
            raise ValueError("cross-MQA dimensions must be 16 aligned")
        self.query_channels = query_channels
        self.context_channels = context_channels
        self.attention_channels = attention_channels
        self.expected_query_tokens = expected_query_tokens
        self.expected_context_tokens = expected_context_tokens
        self.q_conv: nn.Module = nn.Conv2d(
            query_channels, attention_channels, 1, bias=False
        )
        self.q_bn: nn.Module = nn.BatchNorm2d(attention_channels)
        self.kv_conv: nn.Module = nn.Conv2d(
            context_channels, 2 * attention_channels, 1, bias=False
        )
        self.kv_bn: nn.Module = nn.BatchNorm2d(2 * attention_channels)
        self.kv_requant = StaticSymmetricFakeQuant()
        self.channel_scale: Optional[nn.Parameter] = nn.Parameter(
            torch.full(
                (attention_channels,),
                attention_channels**-0.5,
                dtype=torch.float32,
            )
        )
        self.out_conv: nn.Module = nn.Conv2d(
            attention_channels, query_channels, 1, bias=False
        )
        self.out_bn: nn.Module = nn.BatchNorm2d(query_channels)
        self.output_activation = nn.ReLU6(inplace=False)
        self.scale_folded = False

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        batch, _, query_h, query_w = query.shape
        _, _, context_h, context_w = context.shape
        query_tokens = query_h * query_w
        context_tokens = context_h * context_w
        if query_tokens != self.expected_query_tokens:
            raise ValueError("unexpected cross-MQA query token count")
        if context_tokens != self.expected_context_tokens:
            raise ValueError("unexpected cross-MQA context token count")
        q = self.q_bn(self.q_conv(query))
        k, v = self.kv_bn(self.kv_conv(context)).chunk(2, dim=1)
        # MPS matmul backward currently requires the transposed operands to
        # have a compact layout.  Hardware also treats these as explicit
        # matrix descriptors, so make the boundary unambiguous here.
        q = F.relu6(q).flatten(2).transpose(1, 2).contiguous()
        k = F.relu6(k).flatten(2).transpose(1, 2).contiguous()
        v = (
            torch.clamp(v, -6.0, 6.0)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )
        kv = self.kv_requant(
            torch.matmul(k.transpose(-2, -1).contiguous(), v)
        )
        attended = torch.matmul(q, kv)
        # The learned static scale belongs to the attended/output-channel
        # axis.  Placing it here makes it exactly foldable into the input
        # channel axis of out_conv after training.  Scaling q instead is only
        # equivalent while every scale value is identical; learned per-channel
        # values expose that otherwise-hidden deploy mismatch.
        if not self.scale_folded:
            assert self.channel_scale is not None
            attended = attended * (
                self.channel_scale.reshape(1, 1, -1) / float(context_tokens)
            )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, self.attention_channels, query_h, query_w
        )
        projected = self.out_bn(self.out_conv(attended))
        return self.output_activation(query + projected)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.scale_folded:
            return
        assert isinstance(self.q_conv, nn.Conv2d)
        assert isinstance(self.q_bn, nn.BatchNorm2d)
        assert isinstance(self.kv_conv, nn.Conv2d)
        assert isinstance(self.kv_bn, nn.BatchNorm2d)
        assert isinstance(self.out_conv, nn.Conv2d)
        assert isinstance(self.out_bn, nn.BatchNorm2d)
        assert self.channel_scale is not None
        self.q_conv = fuse_conv_bn_eval(self.q_conv, self.q_bn)
        self.q_bn = nn.Identity()
        self.kv_conv = fuse_conv_bn_eval(self.kv_conv, self.kv_bn)
        self.kv_bn = nn.Identity()
        fused_out = fuse_conv_bn_eval(self.out_conv, self.out_bn)
        fused_out.weight.mul_(
            (
                self.channel_scale.detach() / float(self.expected_context_tokens)
            ).reshape(1, -1, 1, 1)
        )
        self.out_conv = fused_out
        self.out_bn = nn.Identity()
        self.register_parameter("channel_scale", None)
        self.scale_folded = True

    def matmul_macs(self, batch: int = 1) -> int:
        width = self.attention_channels
        return batch * (
            self.expected_context_tokens * width * width
            + self.expected_query_tokens * width * width
        )


@dataclass(frozen=True)
class VariantSpec:
    name: str
    summary: str
    channels: Tuple[int, int, int, int]
    block_counts: Tuple[int, int, int, int]
    deploy_grammar: str
    known_non_gemm: Tuple[str, ...]


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "replk_tile": VariantSpec(
        name="replk_tile",
        summary="RepLK-style 5x5/7x7/9x9 dense kernels only at 32x32 or below",
        channels=(32, 64, 96, 128),
        block_counts=(1, 2, 2, 1),
        deploy_grammar="dense 2x2 downsample + partial dense 3/5/7/9x9 + dense 1x1",
        known_non_gemm=("logical partial concat", "decoder addressed resize/add"),
    ),
    "star_pw": VariantSpec(
        name="star_pw",
        summary="StarNet-style fused pointwise gate after reparameterized spatial mixing",
        channels=(32, 64, 96, 128),
        block_counts=(1, 2, 2, 1),
        deploy_grammar="dense 3x3 + fused C-to-2P 1x1 + vector multiply + P-to-C 1x1",
        known_non_gemm=("149504 vector multiplies", "residual adds", "decoder addressed resize/add"),
    ),
    "fullrep_narrow": VariantSpec(
        name="fullrep_narrow",
        summary="Narrow full-channel RepConv network with no partial-channel concat",
        channels=(16, 32, 48, 64),
        block_counts=(1, 2, 2, 2),
        deploy_grammar="full-channel dense 3x3 + dense 2x2 downsample",
        known_non_gemm=("residual-free encoder blocks", "decoder addressed resize/add"),
    ),
    "s2d_matrix_mlp": VariantSpec(
        name="s2d_matrix_mlp",
        summary="Space-to-depth downsampling with an otherwise pointwise matrix-MLP encoder",
        channels=(32, 64, 96, 128),
        block_counts=(2, 2, 3, 3),
        deploy_grammar="dense 2x2 downsample + dense 1x1 residual MLP",
        known_non_gemm=("491520 residual-add elements", "decoder addressed resize/add"),
    ),
    "cross_linear32": VariantSpec(
        name="cross_linear32",
        summary="16x16 queries with fused 8x8 KV and fixed-normalization linear attention",
        channels=(32, 64, 96, 128),
        block_counts=(1, 2, 1, 1),
        deploy_grammar="partial dense 3x3/1x1 + Q/KV 1x1 + two aligned matmuls",
        known_non_gemm=("KTV requant", "cross-scale residual add", "decoder addressed resize/add"),
    ),
    "replk_tile_gemmini_v2": VariantSpec(
        name="replk_tile_gemmini_v2",
        summary=(
            "Direct stride-4 RepStem, consumer-folded 5x5/7x7/9x9 partial "
            "RepLK stages, and a four-scale signed-pair logit pyramid"
        ),
        channels=(16, 32, 48, 64),
        block_counts=(0, 1, 1, 1),
        deploy_grammar=(
            "dense 5x5/s4 stem + three dense 2x2 transitions with folded "
            "mixers + partial dense 5/7/9x9 + four signed-pair vector heads"
        ),
        known_non_gemm=(
            "three split-K descriptor concats",
            "fused signed-pair ReLU6/subtract epilogues",
            "pixel-packed shared-scale logit accumulation",
            "addressed nearest output scatter",
        ),
    ),
}

VARIANT_NAMES: Tuple[str, ...] = tuple(VARIANT_SPECS)


class ArrayPolypVariant(nn.Module):
    """Segmentation shell for the registered hardware exploration variants."""

    def __init__(
        self,
        variant: str,
        num_classes: int = 1,
        input_size: int = 256,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_SPECS:
            raise ValueError(f"unknown variant {variant!r}; choices={VARIANT_NAMES}")
        if input_size % 32:
            raise ValueError("input_size must be divisible by 32")
        self.variant = variant
        self.spec = VARIANT_SPECS[variant]
        self.num_classes = num_classes
        self.input_size = input_size
        c1, c2, c3, c4 = self.spec.channels

        if variant == "replk_tile_gemmini_v2":
            self.down1 = RepStride4Stem(c1)
            self.down2 = nn.Identity()
        else:
            self.down1 = ConvBNAct(3, 16, 3, stride=2)
            self.down2 = SpaceToDepthDownsample(16, c1)
        self.down3 = SpaceToDepthDownsample(c1, c2)
        self.down4 = SpaceToDepthDownsample(c2, c3)
        self.down5 = SpaceToDepthDownsample(c3, c4)

        if variant == "replk_tile":
            self.stage1 = nn.Sequential(ArrayRepBlock(c1, 16))
            self.stage2 = nn.Sequential(
                ArrayRepBlock(c2, 16), RepLKTileBlock(c2, 16, 5)
            )
            self.stage3 = nn.Sequential(
                ArrayRepBlock(c3, 16), RepLKTileBlock(c3, 16, 7)
            )
            self.stage4 = nn.Sequential(RepLKTileBlock(c4, 32, 9))
        elif variant == "star_pw":
            partial = (16, 16, 16, 32)
            gates = (16, 32, 32, 32)
            stages: List[nn.Sequential] = []
            for channels, count, spatial, gate in zip(
                self.spec.channels, self.spec.block_counts, partial, gates
            ):
                stages.append(
                    nn.Sequential(
                        *[
                            StarPWBlock(channels, spatial, gate)
                            for _ in range(count)
                        ]
                    )
                )
            self.stage1, self.stage2, self.stage3, self.stage4 = stages
        elif variant == "fullrep_narrow":
            stages = [
                nn.Sequential(*[RepFullBlock(channels) for _ in range(count)])
                for channels, count in zip(self.spec.channels, self.spec.block_counts)
            ]
            self.stage1, self.stage2, self.stage3, self.stage4 = stages
        elif variant == "s2d_matrix_mlp":
            bottlenecks = (16, 32, 32, 32)
            stages = [
                nn.Sequential(
                    *[
                        PointwiseMatrixMLP(channels, bottleneck)
                        for _ in range(count)
                    ]
                )
                for channels, bottleneck, count in zip(
                    self.spec.channels, bottlenecks, self.spec.block_counts
                )
            ]
            self.stage1, self.stage2, self.stage3, self.stage4 = stages
        elif variant == "cross_linear32":
            self.stage1 = nn.Sequential(ArrayRepBlock(c1, 16))
            self.stage2 = nn.Sequential(
                ArrayRepBlock(c2, 16), ArrayRepBlock(c2, 16)
            )
            self.stage3 = nn.Sequential(ArrayRepBlock(c3, 16))
            self.stage4 = nn.Sequential(ArrayRepBlock(c4, 32))
        elif variant == "replk_tile_gemmini_v2":
            self.stage1 = nn.Identity()
            self.stage2 = nn.Sequential(RepLKConsumerFoldBlock(c2, 16, 5))
            self.stage3 = nn.Sequential(RepLKConsumerFoldBlock(c3, 16, 7))
            self.stage4 = nn.Sequential(RepLKConsumerFoldBlock(c4, 16, 9))
        else:  # pragma: no cover - registry and constructor must stay exhaustive.
            raise AssertionError(f"variant registry has no constructor: {variant}")

        self.cross_attention: Optional[CrossScaleLinearAttention]
        if variant == "cross_linear32":
            self.cross_attention = CrossScaleLinearAttention(
                query_channels=c3,
                context_channels=c4,
                attention_channels=32,
                expected_query_tokens=(input_size // 16) ** 2,
                expected_context_tokens=(input_size // 32) ** 2,
            )
        else:
            self.cross_attention = None

        if variant == "replk_tile_gemmini_v2":
            # The inference decoder works on one-channel logits, not on
            # resized 16-channel features.  Heads 1..3 consume the linear
            # mixer outputs while training; deployment composes each mixer
            # into its head and into the next transition exactly.
            self.scale_heads = nn.ModuleList(
                [
                    nn.Conv2d(channels, 2 * num_classes, 1)
                    for channels in self.spec.channels
                ]
            )
            self.aux_coarse_head = nn.Conv2d(c4, 1, 1)
            self.aux_boundary_head = nn.Conv2d(c1, 1, 1)
            self._consumer_folds_complete = False
        else:
            self.skip1_encode: nn.Module = (
                nn.Identity() if c1 == 16 else ConvBNAct(c1, 16, 1, activation=False)
            )
            self.skip2_encode = ConvBNAct(c2, 16, 1, activation=False)
            self.skip3_encode = ConvBNAct(c3, 16, 1, activation=False)
            self.skip4_encode = ConvBNAct(c4, 16, 1, activation=False)
            self.high_refine = RepFullBlock(16)
            self.semantic_encode = ConvBNAct(16, 16, 1, activation=False)
            self.detail_refine = RepFullBlock(16)
            self.segmentation_head = nn.Conv2d(16, num_classes, 1)
            self.aux_coarse_head = nn.Conv2d(16, 1, 1)
            self.aux_boundary_head = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _resize_like(x: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    @staticmethod
    def _signed_pair_logits(x: Tensor) -> Tensor:
        positive, negative = x.chunk(2, dim=1)
        return F.relu6(positive) - F.relu6(negative)

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size}, got {tuple(x.shape[-2:])}"
            )
        input_size = x.shape[-2:]
        if self.variant == "replk_tile_gemmini_v2":
            x0 = self.stage1(self.down2(self.down1(x)))
            x1 = self.stage2(self.down3(x0))
            x2 = self.stage3(self.down4(x1))
            x3 = self.stage4(self.down5(x2))
            features = (x0, x1, x2, x3)
            logits64 = self._signed_pair_logits(self.scale_heads[0](x0))
            for feature, head in zip(features[1:], self.scale_heads[1:]):
                logits64 = logits64 + F.interpolate(
                    self._signed_pair_logits(head(feature)),
                    size=x0.shape[-2:],
                    mode="nearest",
                )
            logits = F.interpolate(logits64, size=input_size, mode="nearest")
            if not return_aux:
                return logits
            coarse = F.interpolate(
                self.aux_coarse_head(x3), size=input_size, mode="nearest"
            )
            boundary = F.interpolate(
                self.aux_boundary_head(x0), size=input_size, mode="nearest"
            )
            return {
                "segmentation": logits,
                "coarse": coarse,
                "boundary": boundary,
            }

        x1 = self.stage1(self.down2(self.down1(x)))
        skip1 = self.skip1_encode(x1)
        x2 = self.stage2(self.down3(x1))
        skip2 = self.skip2_encode(x2)
        x3 = self.stage3(self.down4(x2))
        x4 = self.stage4(self.down5(x3))
        if self.cross_attention is not None:
            x3 = self.cross_attention(x3, x4)
        skip3 = self.skip3_encode(x3)
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
    def switch_to_deploy(self) -> "ArrayPolypVariant":
        self.eval()
        for module in list(self.modules()):
            if isinstance(module, RepStride4Stem):
                module.switch_to_deploy()
            elif isinstance(module, SpaceToDepthDownsample):
                module.switch_to_deploy()
            elif isinstance(module, RepLargeKernelConv):
                module.switch_to_deploy()
            elif isinstance(module, RepSpatialConv):
                module.switch_to_deploy()
            elif isinstance(module, RepChannelMixer):
                module.switch_to_deploy()
            elif isinstance(module, CrossScaleLinearAttention):
                module.switch_to_deploy()
            elif isinstance(module, FixedLinearAttention):
                module.switch_to_deploy()
        if (
            self.variant == "replk_tile_gemmini_v2"
            and not self._consumer_folds_complete
        ):
            blocks = (self.stage2[0], self.stage3[0], self.stage4[0])
            transitions = (self.down4, self.down5, None)
            for head_index, (block, transition) in enumerate(
                zip(blocks, transitions), start=1
            ):
                if not isinstance(block, RepLKConsumerFoldBlock):
                    raise TypeError("unexpected Gemmini-v2 stage block")
                mixer = block.take_deploy_mixer()
                self.scale_heads[head_index] = _compose_pointwise_before_conv(
                    mixer, self.scale_heads[head_index]
                )
                if transition is not None:
                    if not isinstance(transition, SpaceToDepthDownsample):
                        raise TypeError("unexpected Gemmini-v2 transition")
                    if not isinstance(transition.deploy_conv, nn.Conv2d):
                        raise RuntimeError("transition must be lowered before mixer fold")
                    transition.deploy_conv = _compose_pointwise_before_conv(
                        mixer, transition.deploy_conv
                    )
                if head_index == 3:
                    self.aux_coarse_head = _compose_pointwise_before_conv(
                        mixer, self.aux_coarse_head
                    )
            self._consumer_folds_complete = True
        for module in list(self.modules()):
            if isinstance(module, ConvBNAct) and isinstance(module[1], nn.BatchNorm2d):
                module[0] = fuse_conv_bn_eval(module[0], module[1])
                module[1] = nn.Identity()
        return self

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        auxiliary = sum(
            parameter.numel() for parameter in self.aux_coarse_head.parameters()
        ) + sum(
            parameter.numel() for parameter in self.aux_boundary_head.parameters()
        )
        return {
            "total_parameters": total,
            "active_inference_parameters": total - auxiliary,
            "training_only_aux_parameters": auxiliary,
        }

    def variant_contract(self) -> Dict[str, object]:
        contract: Dict[str, object] = {
            **asdict(self.spec),
            "input_size": self.input_size,
            "array_dim": ARRAY_DIM,
            "forbidden_operators": [
                "depthwise_conv",
                "group_conv",
                "softmax",
                "layer_norm",
                "dynamic_division",
            ],
            "training_dataset": "Kvasir-SEG only",
            "external_inference_only": [
                "ClinicDB",
                "CVC-ColonDB",
                "PolypGen",
                "ETIS-LaribPolypDB",
                "BKAI-IGH NeoPolyp",
            ],
        }
        if self.variant == "replk_tile_gemmini_v2":
            contract.update(
                {
                    "hardware_contract_id": "REPLK_TILE-GEMMINI-V1",
                    "teacher_variant": "replk_tile",
                    "shared_logit_int8_scale_required": True,
                    "capacity_probe_train_dice_gate": 0.95,
                    "formal_kvasir_test_dice_gate": 0.79,
                    "analytical_cycle_gate_at_500mhz": 350_000,
                    "analytical_stretch_cycle_gate_at_500mhz": 250_000,
                    "accuracy_evidence_status": "requires formal Kvasir training",
                }
            )
        return contract

    def extra_hardware_ops(self) -> List[Dict[str, object]]:
        size = self.input_size
        resolutions = (size // 4, size // 8, size // 16, size // 32)
        rows: List[Dict[str, object]] = []
        if self.variant == "star_pw":
            gates = (16, 32, 32, 32)
            for stage, (resolution, gate, count) in enumerate(
                zip(resolutions, gates, self.spec.block_counts), start=1
            ):
                rows.append(
                    {
                        "name": f"stage{stage}.star_multiply",
                        "operator": "elementwise_multiply",
                        "elements": resolution * resolution * gate * count,
                        "target": "vector_alu",
                        "materialized": False,
                    }
                )
                rows.append(
                    {
                        "name": f"stage{stage}.star_product_requant",
                        "operator": "requant",
                        "elements": resolution * resolution * gate * count,
                        "target": "vector_alu",
                        "materialized": False,
                    }
                )
        if self.variant == "s2d_matrix_mlp":
            for stage, (resolution, channels, count) in enumerate(
                zip(resolutions, self.spec.channels, self.spec.block_counts), start=1
            ):
                rows.append(
                    {
                        "name": f"stage{stage}.residual_add",
                        "operator": "elementwise_add",
                        "elements": resolution * resolution * channels * count,
                        "target": "vector_alu",
                        "materialized": False,
                    }
                )
        if self.variant == "replk_tile_gemmini_v2":
            for stage, (resolution, channels) in enumerate(
                zip(resolutions[1:], self.spec.channels[1:]), start=2
            ):
                consumers = [f"scale_heads.{stage - 1}"]
                if stage < 4:
                    consumers.append(f"down{stage + 2}.deploy_conv")
                rows.append(
                    {
                        "name": f"stage{stage}.partial_concat_descriptor",
                        "operator": "concat",
                        "elements": resolution * resolution * channels,
                        "input_elements": resolution * resolution * channels,
                        "output_elements": resolution * resolution * channels,
                        "target": "split_k_address_descriptor",
                        "materialized": False,
                        "lowering": "descriptor",
                        "consumer": consumers,
                        "contract": (
                            "processed 16-channel tile and bypass planes remain "
                            "a logical C16P view; the removed mixer is composed "
                            "into each listed consumer without a memory copy"
                        ),
                    }
                )
        if self.cross_attention is not None:
            width = self.cross_attention.attention_channels
            rows.extend(
                [
                    {
                        "name": "cross_linear32.ktv",
                        "operator": "matmul",
                        "gemm_m": width,
                        "gemm_n": width,
                        "gemm_k": self.cross_attention.expected_context_tokens,
                        "logical_macs": self.cross_attention.expected_context_tokens
                        * width
                        * width,
                        "target": "gemmini_systolic",
                    },
                    {
                        "name": "cross_linear32.qkv",
                        "operator": "matmul",
                        "gemm_m": self.cross_attention.expected_query_tokens,
                        "gemm_n": width,
                        "gemm_k": width,
                        "logical_macs": self.cross_attention.expected_query_tokens
                        * width
                        * width,
                        "target": "gemmini_systolic",
                    },
                    {
                        "name": "cross_linear32.ktv_requant",
                        "operator": "requant",
                        "elements": width * width,
                        "target": "vector_alu",
                        "materialized": True,
                    },
                ]
            )
        return rows


def create_arraypolyp_variant(
    variant: str, **kwargs: object
) -> ArrayPolypVariant:
    return ArrayPolypVariant(variant=variant, **kwargs)
