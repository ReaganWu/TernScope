"""TernScope-Net: SplitLK encoder and the Nearest Unifier decoder.

Module attribute names are stable across FP32 branch fusion and W2A8 loading.
The auxiliary heads belong to the FP32 training topology and are removed from
its 25-convolution deployment graph.
"""
from __future__ import annotations
from typing import Dict, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from .blocks import (
    ConvBNAct, SpaceToDepthDownsample, RepSpatialConv, RepLargeKernelConv,
    RepChannelMixer, SplitLKBlock, SplitLKLargeKernelBlock, RefinementBlock,
)


class TernScopeNet(nn.Module):
    """Four-stage SplitLK encoder with a 16-channel Nearest Unifier decoder."""

    def __init__(self, num_classes: int = 1, input_size: int = 256) -> None:
        super().__init__()
        if input_size % 32:
            raise ValueError("input_size must be divisible by 32")
        self.num_classes = num_classes
        self.input_size = input_size
        c1, c2, c3, c4 = 32, 64, 96, 128
        self.down1 = ConvBNAct(3, 16, 3, stride=2)
        self.down2 = SpaceToDepthDownsample(16, c1)
        self.down3 = SpaceToDepthDownsample(c1, c2)
        self.down4 = SpaceToDepthDownsample(c2, c3)
        self.down5 = SpaceToDepthDownsample(c3, c4)
        self.stage1 = nn.Sequential(SplitLKBlock(c1, 16))
        self.stage2 = nn.Sequential(SplitLKBlock(c2, 16), SplitLKLargeKernelBlock(c2, 16, 5))
        self.stage3 = nn.Sequential(SplitLKBlock(c3, 16), SplitLKLargeKernelBlock(c3, 16, 7))
        self.stage4 = nn.Sequential(SplitLKLargeKernelBlock(c4, 32, 9))
        # Nearest Unifier: four 16-channel projections and semantic/detail fusion.
        self.skip1_encode = ConvBNAct(c1, 16, 1, activation=False)
        self.skip2_encode = ConvBNAct(c2, 16, 1, activation=False)
        self.skip3_encode = ConvBNAct(c3, 16, 1, activation=False)
        self.skip4_encode = ConvBNAct(c4, 16, 1, activation=False)
        self.high_refine = RefinementBlock(16)
        self.semantic_encode = ConvBNAct(16, 16, 1, activation=False)
        self.detail_refine = RefinementBlock(16)
        self.segmentation_head = nn.Conv2d(16, num_classes, 1)
        self.aux_coarse_head = nn.Conv2d(16, 1, 1)
        self.aux_boundary_head = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _resize_like(x: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f'expected {self.input_size}x{self.input_size}, got {tuple(x.shape[-2:])}')
        input_size = x.shape[-2:]
        x1 = self.stage1(self.down2(self.down1(x)))
        skip1 = self.skip1_encode(x1)
        x2 = self.stage2(self.down3(x1))
        skip2 = self.skip2_encode(x2)
        x3 = self.stage3(self.down4(x2))
        x4 = self.stage4(self.down5(x3))
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
    def switch_to_deploy(self) -> "TernScopeNet":
        self.eval()
        for module in list(self.modules()):
            if isinstance(
                module,
                (SpaceToDepthDownsample, RepLargeKernelConv, RepSpatialConv, RepChannelMixer),
            ):
                module.switch_to_deploy()
        for module in list(self.modules()):
            if isinstance(module, ConvBNAct) and isinstance(module[1], nn.BatchNorm2d):
                module[0] = fuse_conv_bn_eval(module[0], module[1])
                module[1] = nn.Identity()
        return self

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        auxiliary = sum(
            parameter.numel()
            for head in (self.aux_coarse_head, self.aux_boundary_head)
            for parameter in head.parameters()
        )
        return {
            "total_parameters": total,
            "active_inference_parameters": total - auxiliary,
            "training_only_aux_parameters": auxiliary,
        }


def create_ternscope(**kwargs: object) -> TernScopeNet:
    """Construct the published TernScope-Net FP32 training topology."""
    return TernScopeNet(**kwargs)
