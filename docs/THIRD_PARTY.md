# Third-party sources

The root Apache-2.0 license covers the TernScope project code. Included derived memory-simulation files retain their upstream license terms.

`hardware/sim/memory/{SimDRAM_ternscope.cc,mm.cc,mm.h,mm_dramsim2.cc,mm_dramsim2.h,host_memory.cc}` originate from the supplied testchipip / Rocket Chip simulation integration and its TernScope modifications. The DRAMSim2 adapter includes the corrected DDR response handling and backend checks. Original in-file notices are preserved.

- [testchipip](https://github.com/ucb-bar/testchipip), BSD-3-Clause: [license text](../licenses/testchipip-BSD-3-Clause.txt).
- [Rocket Chip](https://github.com/chipsalliance/rocket-chip), Berkeley BSD notice: [license text](../licenses/rocket-chip-BSD.txt).

Verilator, Icarus Verilog, Yosys/ABC, DRAMSim2, fesvr, standard-cell libraries, and datasets are external dependencies. Their source trees, binaries, library data, and datasets are not redistributed here. Baseline segmentation adapters are omitted from this focused release.
