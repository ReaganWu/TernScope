# TernScope accelerator

The TernScope accelerator executes the W2A8 deployment graph described in [the paper](https://arxiv.org/abs/2610.29617). A **16 × 16 ternary PE array** performs convolution, and the shared **Affine & Quantization Engine** handles per-channel scaling, bias, feature addition, rounding, and saturation.

| Source | Role |
| --- | --- |
| `rtl/ternscope_accelerator_top.sv` | Accelerator top level |
| `rtl/ternscope_array.sv` | Ternary PE array |
| `rtl/shared_engine.sv` | Shared affine and quantization datapath |
| `rtl/ternscope_core.sv` | Workload controller and on-chip dataflow |
| `rtl/ternscope_mem.sv` | Feature, weight, and lookup-table storage |
| `rtl/ternscope_dma.sv`, `rtl/ternscope_row_cdc.sv` | Memory transfers and clock-domain crossing |

## Module checks

With Icarus Verilog (`iverilog` and `vvp`) on PATH:

```bash
python hardware/tests/run.py
```

The tests exercise ternary dot products and signed extrema, tags and backpressure, round-to-nearest-even, saturation, two-source Add, and CDC ordering. Compiled binaries stay in `build/rtl-tests/`.

## Generate the checkpoint-specific workload

The control schedule is fixed to the published 25-convolution / 3-Add topology. Generate the scales, biases, and lookup tables from the selected checkpoint.

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

The bundled checkpoint needs **27 LUT tables (6,912 entries)**. The RTL default is **21 tables (5,376 entries)**; the build script reads the generated control configuration and sets `LUT_ENTRIES` explicitly. A different ROM capacity changes the implementation configuration. Paper area and cycle results are not new measurements of this checkpoint-specific build.

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
../dramsim2/Vternscope_system +dramsim \
  +dramsim_ini_dir=/absolute/path/to/testchipip/src/main/resources/dramsim2_ini \
  +data=../vectors +frames=8
```

The driver checks every node and final output against the generated integer oracle and requires the actual DRAMSim2 backend. The simulated core/memory clocks are 100/500 MHz. `--micro` on the builder selects the existing DDR round-trip test instead of the full-network driver.

## Logic synthesis

```bash
python hardware/synthesis/run.py --liberty path/to/library.lib \
  --control build/control --output build/synthesis
```

The script synthesizes the supplied RTL using Yosys/ABC and the selected Liberty library. Mutable SRAMs are represented as black boxes; ROM logic is retained. SRAM macro area, CPU, external DDR, clock-tree synthesis, and routing are outside this scope.

The paper's Table III uses a matched Nangate45 comparison against the original Gemmini. Reproducing that table requires the corresponding baseline, constraints, memory accounting, and power-analysis setup. The matched comparison and OpenSTA power-analysis scripts are not included here; this command reports the supplied accelerator configuration.

The memory-simulation adapters retain their upstream notices; see [third-party sources](../docs/THIRD_PARTY.md).
