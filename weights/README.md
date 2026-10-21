# Pretrained W2A8 model

`ternscope_w2a8.pth` is the single default inference checkpoint: all 25 convolutions use ternary weights and 8-bit activations.

The supplied checkpoint for Progressive QAT. Its model tensors are preserved; optimizer state, RNG state, and training history have been removed. The file retains the original training recipe and epoch as metadata.

Use it with `software/infer.py`, `software/evaluate.py`, or the hardware integer-reference exporter. This is a fake-quantized software model; integer export and RTL simulation are separate validation steps. This file is not a Progressive training-resume checkpoint.
