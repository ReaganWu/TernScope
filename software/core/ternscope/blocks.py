"""Re-parameterizable blocks used by TernScope-Net."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

ARRAY_DIM = 16


def _activation(enabled: bool = True) -> nn.Module:
    # Bound activations before the deployment quantization boundaries.
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


def _aligned(value: int) -> bool:
    return value % ARRAY_DIM == 0


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


class SplitLKBlock(nn.Module):
    """SplitLK spatial mixing on one or two 16-channel array tiles.

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


class RefinementBlock(nn.Module):
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


class SplitLKLargeKernelBlock(nn.Module):
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
