"""Run self-contained T300 module checks; generated binaries stay outside sources."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('build/rtl-tests'))
    args=parser.parse_args()
    for tool in ('iverilog','vvp'):
        if not shutil.which(tool):
            parser.error(f'{tool} is required on PATH.')
    hardware=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    cases={
        'tb_array':['rtl/t300_array13.sv','tests/tb_array.sv'],
        'tb_affine':['rtl/shared_engine.sv','tests/tb_affine.sv'],
        'tb_cdc':['rtl/async_fifo.sv','rtl/t300_row_cdc.sv','tests/tb_cdc.sv'],
    }
    for name,files in cases.items():
        binary=(args.output/(name+'.vvp')).resolve()
        subprocess.run(['iverilog','-g2012','-s',name,'-o',str(binary),*[str(hardware/f) for f in files]],check=True)
        subprocess.run(['vvp',str(binary)],check=True,timeout=60)


if __name__=='__main__':
    main()
