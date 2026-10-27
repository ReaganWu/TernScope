"""Train TernScope-Net with the 200-epoch TPQ recipe and explicit local paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Kvasir-SEG directory with images/ and masks/.')
    parser.add_argument('--fp32', type=Path, required=True, help='FP32 training-topology checkpoint for initialization.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0', help='One CUDA device index or UUID.')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true', help='Fold FP32, initialize scales, and check TPQ initialization and quantization transitions.')
    action.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if ',' in args.gpu or not args.gpu.strip():
        parser.error('--gpu must identify one CUDA device.')
    if not args.fp32.is_file():
        parser.error('--fp32 must point to a local FP32 training-topology checkpoint.')
    software = Path(__file__).resolve().parent
    split = json.loads((software/'data/kvasir_split.json').read_text())
    missing = [str(args.data/d/i) for d in ('images','masks') for ids in split.values() for i in ids if not (args.data/d/i).is_file()]
    if missing:
        parser.error(f'Missing {len(missing)} dataset files; first: {missing[0]}')
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TERNSCOPE_RUN_DIR=str(args.output.resolve()),
               TERNSCOPE_DATA_ROOT=str(args.data.resolve()), TERNSCOPE_FP32=str(args.fp32.resolve()),
               CUDA_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE='1')
    script = software/'training/tpq'/('prepare.py' if args.prepare else 'train.py')
    command = [sys.executable, str(script)]
    if not args.prepare:
        command += ['--batch','2','--gpu',args.gpu]
        if args.resume:
            command.append('--resume')
    result = subprocess.run(command, env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
