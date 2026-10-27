"""Dataset, augmentation, and segmentation metrics for TernScope-Net."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegmentationSample:
    dataset: str
    image: str
    mask: str
    sample_id: str
    sequence: Optional[str] = None


class PolypSegmentationDataset(Dataset[Tuple[Tensor, Tensor, str, str, str]]):
    """PIL-only paired loader; no torchvision/albumentations dependency."""

    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.2:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotations = random.randrange(4)
        transpose = {
            1: Image.Transpose.ROTATE_90,
            2: Image.Transpose.ROTATE_180,
            3: Image.Transpose.ROTATE_270,
        }
        if rotations:
            image = image.transpose(transpose[rotations])
            mask = mask.transpose(transpose[rotations])
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.12:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))
        return image, mask

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str, str, str]:
        sample = self.samples[index]
        with Image.open(sample.image) as source_image:
            image = source_image.convert("RGB")
        with Image.open(sample.mask) as source_mask:
            mask = source_mask.convert("L")
        if image.size != mask.size:
            raise ValueError(f"image/mask size mismatch for {sample.sample_id}")
        if self.augment:
            image, mask = self._augment(image, mask)
        size = (self.image_size, self.image_size)
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).copy()
        mask_array = np.asarray(mask, dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1) / 127.5 - 1.0
        mask_tensor = torch.from_numpy((mask_array > 127).astype(np.float32)).unsqueeze(0)
        return (
            image_tensor,
            mask_tensor,
            sample.dataset,
            sample.sample_id,
            sample.sequence or "",
        )


def _boundary(mask: Tensor, kernel_size: int = 3) -> Tensor:
    padding = kernel_size // 2
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded) > 0.5


def _boundary_f1(prediction: Tensor, target: Tensor, tolerance: int = 2) -> Tensor:
    pred_boundary = _boundary(prediction)
    target_boundary = _boundary(target)
    kernel = tolerance * 2 + 1
    pred_dilated = F.max_pool2d(pred_boundary.float(), kernel, stride=1, padding=tolerance) > 0
    target_dilated = F.max_pool2d(target_boundary.float(), kernel, stride=1, padding=tolerance) > 0
    dims = (1, 2, 3)
    pred_count = pred_boundary.sum(dims).float()
    target_count = target_boundary.sum(dims).float()
    precision = (pred_boundary & target_dilated).sum(dims).float() / pred_count.clamp_min(1)
    recall = (target_boundary & pred_dilated).sum(dims).float() / target_count.clamp_min(1)
    score = 2 * precision * recall / (precision + recall).clamp_min(1e-7)
    both_empty = (pred_count == 0) & (target_count == 0)
    return torch.where(both_empty, torch.ones_like(score), score)


class SegmentationMetrics:
    """Macro, pooled, boundary and empty-frame metrics with explicit semantics."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.count = 0
        self.sums: Dict[str, float] = {
            "dice": 0.0,
            "iou": 0.0,
            "specificity": 0.0,
            "boundary_f1": 0.0,
            "mae": 0.0,
        }
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.positive_count = 0
        self.positive_sensitivity = 0.0
        self.positive_dice = 0.0
        self.positive_iou = 0.0
        self.positive_boundary_f1 = 0.0
        self.small_count = 0
        self.small_sensitivity = 0.0
        self.empty_count = 0
        self.empty_false_positive_count = 0
        self.empty_predicted_area_fractions: List[float] = []
        self.sequence: Dict[str, Dict[str, float]] = {}

    @torch.no_grad()
    def update(
        self,
        logits: Tensor,
        target: Tensor,
        sequences: Optional[Sequence[str]] = None,
    ) -> None:
        probability = logits.sigmoid()
        prediction = probability >= self.threshold
        truth = target >= 0.5
        dims = (1, 2, 3)
        tp = (prediction & truth).sum(dims).float()
        fp = (prediction & ~truth).sum(dims).float()
        fn = (~prediction & truth).sum(dims).float()
        tn = (~prediction & ~truth).sum(dims).float()
        target_area = truth.sum(dims).float()
        pred_area = prediction.sum(dims).float()
        smooth = 1.0
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        positive = target_area > 0
        sensitivity = torch.where(
            positive,
            tp / (tp + fn).clamp_min(1),
            torch.zeros_like(tp),
        )
        specificity = tn / (tn + fp).clamp_min(1)
        boundary = _boundary_f1(prediction, truth)
        mae = (probability - target).abs().mean(dims)
        batch = logits.shape[0]
        self.count += batch
        for name, values in (
            ("dice", dice),
            ("iou", iou),
            ("specificity", specificity),
            ("boundary_f1", boundary),
            ("mae", mae),
        ):
            self.sums[name] += float(values.sum().item())
        self.positive_count += int(positive.sum().item())
        self.positive_sensitivity += float(sensitivity[positive].sum().item())
        self.positive_dice += float(dice[positive].sum().item())
        self.positive_iou += float(iou[positive].sum().item())
        self.positive_boundary_f1 += float(boundary[positive].sum().item())
        self.tp += float(tp.sum().item())
        self.fp += float(fp.sum().item())
        self.fn += float(fn.sum().item())
        self.tn += float(tn.sum().item())
        small = positive & (target_area < 0.01 * target[0].numel())
        self.small_count += int(small.sum().item())
        self.small_sensitivity += float(sensitivity[small].sum().item())
        empty = target_area == 0
        self.empty_count += int(empty.sum().item())
        self.empty_false_positive_count += int((empty & (pred_area > 0)).sum().item())
        pixels_per_frame = float(target[0].numel())
        self.empty_predicted_area_fractions.extend(
            (pred_area[empty] / pixels_per_frame).detach().cpu().tolist()
        )
        if sequences:
            for index, sequence_name in enumerate(sequences):
                if not sequence_name:
                    continue
                values = self.sequence.setdefault(
                    sequence_name,
                    {
                        "count": 0.0,
                        "dice": 0.0,
                        "positive_count": 0.0,
                        "positive_dice": 0.0,
                        "empty_count": 0.0,
                        "empty_false_positive_count": 0.0,
                    },
                )
                values["count"] += 1.0
                values["dice"] += float(dice[index].item())
                if bool(positive[index].item()):
                    values["positive_count"] += 1.0
                    values["positive_dice"] += float(dice[index].item())
                else:
                    values["empty_count"] += 1.0
                    values["empty_false_positive_count"] += float(pred_area[index].item() > 0)

    def compute(self) -> Dict[str, object]:
        if self.count == 0:
            raise RuntimeError("no metric samples accumulated")
        pooled_dice = (2 * self.tp + 1.0) / (2 * self.tp + self.fp + self.fn + 1.0)
        pooled_iou = (self.tp + 1.0) / (self.tp + self.fp + self.fn + 1.0)
        averaged = {
            name: value / self.count
            for name, value in self.sums.items()
        }
        empty_area_mean = None
        empty_area_p95 = None
        if self.empty_predicted_area_fractions:
            empty_area_mean = float(np.mean(self.empty_predicted_area_fractions))
            empty_area_p95 = float(np.quantile(self.empty_predicted_area_fractions, 0.95))
        result: Dict[str, object] = {
            "count": self.count,
            **averaged,
            "sensitivity": (
                self.positive_sensitivity / self.positive_count
                if self.positive_count
                else None
            ),
            "positive_frame_count": self.positive_count,
            "nonempty_dice": (
                self.positive_dice / self.positive_count if self.positive_count else None
            ),
            "nonempty_iou": (
                self.positive_iou / self.positive_count if self.positive_count else None
            ),
            "nonempty_boundary_f1": (
                self.positive_boundary_f1 / self.positive_count
                if self.positive_count
                else None
            ),
            "pooled_dice": pooled_dice,
            "pooled_iou": pooled_iou,
            "small_lesion_count": self.small_count,
            "small_lesion_sensitivity": (
                self.small_sensitivity / self.small_count if self.small_count else None
            ),
            "empty_frame_count": self.empty_count,
            "empty_frame_false_positive_rate": (
                self.empty_false_positive_count / self.empty_count if self.empty_count else None
            ),
            "empty_predicted_area_fraction_mean": empty_area_mean,
            "empty_predicted_area_fraction_p95": empty_area_p95,
        }
        if self.sequence:
            result["sequence_macro_dice"] = {
                name: values["dice"] / values["count"]
                for name, values in sorted(self.sequence.items())
            }
            result["sequence_nonempty_macro_dice"] = {
                name: (
                    values["positive_dice"] / values["positive_count"]
                    if values["positive_count"]
                    else None
                )
                for name, values in sorted(self.sequence.items())
            }
            result["sequence_empty_frame_false_positive_rate"] = {
                name: (
                    values["empty_false_positive_count"] / values["empty_count"]
                    if values["empty_count"]
                    else None
                )
                for name, values in sorted(self.sequence.items())
            }
            result["sequence_unweighted_macro_dice"] = float(
                np.mean(
                    [values["dice"] / values["count"] for values in self.sequence.values()]
                )
            )
        return result
