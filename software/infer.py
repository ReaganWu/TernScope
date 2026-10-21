"""Predict a 256x256 binary mask from a fully W2A8 checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from runtime import configure, image_tensor, load_w2a8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parents[1]/'weights/ternscope_w2a8.pth')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--trusted-checkpoint', action='store_true')
    args = parser.parse_args()
    configure(args.device)
    model = load_w2a8(args.checkpoint, args.device, args.trusted_checkpoint)
    with torch.inference_mode():
        logits = model(image_tensor(args.image).unsqueeze(0).to(args.device))
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite predictions.')
    mask = (logits.sigmoid()[0, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        Image.fromarray(mask).save(handle, format='PNG')
    print(f'Saved 256x256 mask: {args.output}')


if __name__ == '__main__':
    main()
