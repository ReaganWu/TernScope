# TernScope-Net and TPQ

This directory implements **TernScope-Net** and **Ternary Progressive Quantization-aware Training (TPQ)** from [the paper](https://arxiv.org/abs/2610.29617), with inference and evaluation on CPU or CUDA.

| Component | Source |
| --- | --- |
| TernScope-Net: SplitLK encoder and Nearest Unifier decoder | `core/ternscope/model.py` |
| Re-parameterizable convolution blocks and branch fusion | `core/ternscope/blocks.py` |
| Branch-fused deployment topology | `core/topology.py` (`TernScopeDeploy`) |
| Ternary weight and 8-bit activation quantization | `core/quantization.py` |
| TPQ initialization and training schedule | `training/tpq/prepare.py`, `training/tpq/train.py` |

`TernScopeNet` / `create_ternscope()` construct the FP32 training topology. Branch fusion yields the 25-convolution deployment graph; TPQ adapts that graph to W2A8. Module attribute names and tensor keys are preserved so the supplied checkpoints load strictly.

## Install and check

From the repository root, with Python 3.9 or newer:

```bash
python -m pip install -r software/requirements.txt
python software/smoke.py
```

The smoke checks need no dataset or pretrained weights. They cover branch fusion, the 25-convolution deployment graph, ternary ranges, gradients, and frozen model reload.

## Inference

```bash
python software/infer.py --image path/to/image.png --output runs/mask.png
```

This uses `weights/ternscope_w2a8.pth` and saves a **256 × 256** mask. RGB input is resized with PIL bilinear interpolation and normalized to [−1, 1]. Prediction uses sigmoid ≥ 0.5. Add `--device cuda` to use a GPU or `--checkpoint` to select another compatible W2A8 model.

## Evaluation

Obtain dataset images and masks from the original providers. Do not redistribute datasets with the code. Arrange each dataset as `images/` and `masks/`; retain the original Kvasir filenames.

```bash
python software/evaluate.py \
  --kvasir data/Kvasir-SEG \
  --clinicdb data/CVC-ClinicDB \
  --colondb data/CVC-ColonDB \
  --output runs/evaluation
```

Kvasir uses the supplied fixed split in `data/kvasir_split.json`: 720 training, 80 validation, and 200 test samples. ClinicDB and ColonDB use 612 and 380 images. External datasets are optional arguments. The evaluator writes per-image Dice, IoU, and Boundary F1 plus their macro means; Dice/IoU use smoothing 1. Masks use nearest-neighbor resizing and threshold >127. Use `--split validation` for the fixed validation set.

## TPQ: Ternary Progressive Quantization-aware Training

TPQ starts from an **FP32 training-topology checkpoint**, folds its branches, adapts all layers in W8A8, and converts groups to W2A8. The FP32 initializer is an external input; the bundled W2A8 inference weight is not an FP32 initializer or a training-resume checkpoint.

```bash
python software/train.py \
  --data data/Kvasir-SEG --fp32 path/to/fp32.pth \
  --output runs/tpq --gpu 0 --prepare

python software/train.py \
  --data data/Kvasir-SEG --fp32 path/to/fp32.pth \
  --output runs/tpq --gpu 0
```

Add `--resume` to the second command to continue that run's `bs2/last.pth`. Preparation checks, initial state, and training outputs stay under the selected output directory. CUDA is required for this training recipe. Batch size is 2, training lasts 200 epochs, and no distillation is used.

| Epoch | Weight conversion |
| --- | --- |
| 1–10 | All convolutions W8A8 |
| 11 / 21 / 31 / 41 | Deep-to-shallow encoder groups |
| 51 | Skip encoders |
| 61 / 71 | Decoder groups |
| 81 | Stem and prediction head; all 25 convolutions now W2A8 |
| 81–200 | Fully W2A8 training and checkpoint selection |

Each transition recalibrates on the first 64 deterministically augmented training examples and caps the learning rate for three epochs. Activation observers follow the supplied training schedule, with final freezing after epoch 83. Validation Dice selects the final checkpoint only from epochs 81–200.

The supplied training environment recorded PyTorch 2.4.1+cu121. Local CPU validation of this portable package used PyTorch 2.13.0, NumPy 2.5.2, and Pillow 12.3.0. See [validation](../docs/VALIDATION.md) for checks run on this release. For an original trusted training-state checkpoint containing legacy Python/NumPy objects, inference and evaluation accept `--trusted-checkpoint`; the bundled tensor-only weight does not need it.
