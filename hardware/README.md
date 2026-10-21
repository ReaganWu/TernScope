# TernScope accelerator

`rtl/` contains the final T300 datapath: the 16 × 16 ternary array, shared affine engine, feature/source memories, controller, DMA, and clock-domain crossing. Earlier T100/T200 overlays, prototype controllers, failed FPGA attempts, physical-design experiments, and generated netlists are omitted.

## Module checks

With Icarus Verilog (`iverilog` and `vvp`) on PATH:

```bash
python hardware/tests/run.py
```

The tests exercise ternary dot products and signed extrema, tags and backpressure, round-to-nearest-even, saturation, two-source Add, and CDC ordering. Compiled binaries stay in `build/rtl-tests/`.

## Generate the checkpoint-specific workload

The control schedule is fixed to the published 25-convolution / 3-Add topology. Numeric tables must be generated from the selected checkpoint; old checkpoint scales and biases are not bundled.

```bash
python hardware/reference/export_frozen_graph.py \
  --synthetic --output build/integer-graph

python hardware/reference/prepare_data.py \
  --artifacts build/integer-graph --out build/vectors

python hardware/reference/compile_control.py \
  --artifacts build/integer-graph --output build/control
```

The default checkpoint is the bundled W2A8 weight. Synthetic mode produces four deterministic inputs for arithmetic checks and makes no dataset-accuracy claim. To use the first four fixed Kvasir validation samples, replace `--synthetic` with `--data-root data/Kvasir-SEG`.

Export produces a NumPy integer oracle with packed W2 codes, Q24 coefficients, bias, and edge LUTs. The preparation step independently checks all 28 node accumulators and A8 outputs and emits the RTL vectors. The control compiler checks topology and bounded arithmetic before writing the ROMs.

The bundled checkpoint needs **27 LUT tables (6,912 entries)**. The core's original default remains **21 tables (5,376 entries)**; the build script reads the generated control configuration and sets `LUT_ENTRIES` explicitly. A different ROM capacity changes the implementation configuration. Paper area and cycle results are not new measurements of this checkpoint-specific build.

Fixed-point export can differ from software fake quantization. `reference_report.json` records these differences; successful integer-oracle generation does not establish RTL execution or bit-exact agreement with the floating-point graph.

## RTL + DRAMSim2

External dependencies are **Verilator**, **testchipip**, **DRAMSim2**, and a RISC-V prefix providing **fesvr** headers and `libfesvr.a`. Install/build those separately; they are not vendored into this repository.

```bash
python hardware/sim/build.py \
  --testchipip path/to/testchipip --dramsim2 path/to/DRAMSim2 \
  --fesvr path/to/riscv-prefix --control build/control \
  --output build/dramsim2
```

Run the generated binary with the control directory as the working directory, so `$readmemh` finds its ROMs:

```bash
cd build/control
../dramsim2/Vt300_system_500 +dramsim \
  +dramsim_ini_dir=/absolute/path/to/testchipip/src/main/resources/dramsim2_ini \
  +data=../vectors +frames=8
```

The driver checks every node and final output against the generated integer oracle and requires the actual DRAMSim2 backend. The simulated core/memory clocks are 100/500 MHz. `--micro` on the builder selects the existing DDR round-trip test instead of the full-network driver.

## Logic synthesis

```bash
python hardware/synthesis/run.py --liberty path/to/library.lib \
  --control build/control --output build/synthesis
```

This is the portable **historical phase-7** flow with mutable SRAMs represented as blackboxes and ROM logic retained. It is useful for building the design, but the delivery did not contain the later matched original-Gemmini/TernScope comparison flow used for the paper's 88.46%/70.28% figures. Those figures cannot be reproduced by treating this historical flow as the matched comparison. SRAM macro area, CPU, DDR, clock-tree synthesis, and routing are outside this logic-synthesis scope.

The memory-simulation adapters retain their upstream notices; see [third-party sources](../docs/THIRD_PARTY.md).
