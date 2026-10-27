"""Build the corrected DRAMSim2 harness using externally installed dependencies."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--testchipip',type=Path,required=True,help='External testchipip repository root.')
    parser.add_argument('--dramsim2',type=Path,required=True)
    parser.add_argument('--fesvr',type=Path,required=True,help='RISC-V installation prefix with include/fesvr and lib/libfesvr.a.')
    parser.add_argument('--output',type=Path,default=Path('build/dramsim2'))
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--micro',action='store_true')
    parser.add_argument('--control',type=Path,help='Generated control directory (required for full-network build).')
    args=parser.parse_args()
    hardware=Path(__file__).resolve().parents[1]
    sim=hardware/'sim';memory=sim/'memory'
    support=args.testchipip.resolve()/'src/main/resources/testchipip'
    dramsim=args.dramsim2.resolve();fesvr=args.fesvr.resolve()
    required=[support/'vsrc/SimDRAM.v',support/'csrc',dramsim/'libdramsim.a',fesvr/'lib/libfesvr.a',fesvr/'include/fesvr']
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        parser.error('Missing external dependencies: '+', '.join(missing))
    if not shutil.which('verilator'):
        parser.error('verilator is required on PATH.')
    if args.jobs<1:
        parser.error('--jobs must be positive.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.micro:
        mode=['--binary','--timing'];top='tb_ddr'
        sources=[hardware/'rtl/ternscope_dma.sv',sim/'ternscope_ddr_harness.sv',sim/'tb_ddr.sv']
    else:
        if args.control is None:parser.error('--control is required for full-network build.')
        config=json.loads((args.control/'config.json').read_text())
        mode=['--cc','--exe','--public-flat-rw'];top='ternscope_system'
        mode += ['-GLUT_ENTRIES='+str(config['lut_entries'])]
        sources=[hardware/'rtl'/n for n in ['ternscope_mem.sv','ternscope_array.sv','acc_row_fifo.sv','shared_engine.sv','ternscope_core.sv','ternscope_dma.sv','async_fifo.sv','ternscope_row_cdc.sv']]
        sources += [sim/'ternscope_ddr_harness.sv',sim/'ternscope_system.sv',sim/'simulate.cpp']
    sources += [support/'vsrc/SimDRAM.v']+[memory/n for n in ['SimDRAM_ternscope.cc','mm.cc','mm_dramsim2.cc','host_memory.cc']]
    flags=' '.join(shlex.quote('-I'+str(p)) for p in [memory,support/'csrc',dramsim,fesvr/'include'])
    libraries=' '.join(shlex.quote(str(p)) for p in [dramsim/'libdramsim.a',fesvr/'lib/libfesvr.a'])+' -lpthread'
    command=['verilator',*mode,'--build','--vpi','--top-module',top,'-Wno-fatal','-j',str(args.jobs),'--Mdir',str(output),*[str(p) for p in sources],'-CFLAGS',flags,'-LDFLAGS',libraries]
    subprocess.run(command,check=True)
    print('Built:',output/('V'+top))
    print('Run full-network simulation from the generated control directory; pass +data and +dramsim_ini_dir explicitly.')


if __name__=='__main__':
    main()
