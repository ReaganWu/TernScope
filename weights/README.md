# Pretrained W2A8 model

`ternscope_w2a8.pth` is the single default inference checkpoint: all 25 convolutions use ternary weights and 8-bit activations.

The release preserves the supplied model tensors and their original training-recipe and epoch metadata. Optimizer state, RNG state, and training history are excluded.

Use it with `software/infer.py`, `software/evaluate.py`, or the hardware integer-reference exporter. This is a fake-quantized software model; integer export and RTL simulation are separate validation steps. This file is not a TPQ training-resume checkpoint.
