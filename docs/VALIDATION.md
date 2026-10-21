# Curation validation

Local checks on 2026-10-21:

- Model checks passed: FP32 branch fusion, removal of unused auxiliary heads from the deployment graph, 25 ternary convolutions, 53 activation boundaries, finite gradients, and exact frozen-model reload.
- The bundled checkpoint's model tensors equal the supplied epoch146 checkpoint. CPU outputs were bit-exact to the original source snapshot on two RGB 256 × 256 inputs; inference did not change model buffers.
- The inference CLI saved a binary mask. The evaluation CLI processed an 80-image synthetic fixture and wrote per-image and aggregate outputs. This was a plumbing test, not a medical segmentation benchmark.
- Integer export and independent oracle preparation passed for four deterministic synthetic inputs across all 28 nodes. Control generation produced 27 LUT tables and passed coefficient and conservative ternary-accumulation width checks.
- Slang 12.0 parsed/elaborated the T300 RTL and the three module testbenches without errors. All Python files passed syntax checks; documentation links and image references resolve locally.

This curation did not rerun CUDA training, Icarus event simulation, full-network RTL/DRAMSim2 simulation, synthesis, physical implementation, or medical-dataset evaluation. The integer oracle differs from fake-quantized software on some synthetic samples; it is not presented as a bit-exact replacement for the software predictions.

CPU environment: PyTorch 2.13.0, NumPy 2.5.2, Pillow 12.3.0. Original training records specify PyTorch 2.4.1+cu121.
