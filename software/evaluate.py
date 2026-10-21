"""Fixed-split, per-image W2A8 evaluation. No checkpoint selection or recalibration."""
import argparse
import csv
import json
from pathlib import Path
import torch
from runtime import (SOFTWARE, PolypSegmentationDataset, SegmentationSample,
                     SegmentationMetrics, configure, fixed_split, load_w2a8, sha256)


def external(name, root, count):
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = sorted(p for p in (root / 'images').iterdir() if p.suffix.lower() in suffixes)
    masks = [p for p in (root / 'masks').iterdir() if p.suffix.lower() in suffixes]
    by_stem = {p.stem: p for p in masks}
    if len(images) != count or len(masks) != count or len(by_stem) != count:
        raise ValueError(f'{name}: expected exactly {count} paired images and masks.')
    return [SegmentationSample(name, p, by_stem[p.stem], p.name) for p in images]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parents[1]/'weights/ternscope_w2a8.pth')
    parser.add_argument('--kvasir', type=Path, required=True, help='Directory containing images/ and masks/.')
    parser.add_argument('--clinicdb', type=Path)
    parser.add_argument('--colondb', type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--trusted-checkpoint', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    split = fixed_split()
    sets = {f'Kvasir_{args.split}': [SegmentationSample('Kvasir', args.kvasir/'images'/i, args.kvasir/'masks'/i, i) for i in split[args.split]]}
    if args.clinicdb:
        sets['CVC-ClinicDB'] = external('CVC-ClinicDB', args.clinicdb, 612)
    if args.colondb:
        sets['CVC-ColonDB'] = external('CVC-ColonDB', args.colondb, 380)
    configure(args.device)
    model = load_w2a8(args.checkpoint, args.device, args.trusted_checkpoint)
    checkpoint_hash = sha256(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, summaries = [], []
    with torch.inference_mode():
        for name, samples in sets.items():
            dataset = PolypSegmentationDataset(samples, 256, False)
            current = []
            for start in range(0, len(dataset), args.batch_size):
                batch = [dataset[i] for i in range(start, min(start+args.batch_size, len(dataset)))]
                logits = model(torch.stack([b[0] for b in batch]).to(args.device)).cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError(f'Non-finite predictions in {name}.')
                for j, item in enumerate(batch):
                    metric = SegmentationMetrics(0.5)
                    metric.update(logits[j:j+1], item[1].unsqueeze(0))
                    scores = metric.compute()
                    current.append(dict(dataset=name, id=samples[start+j].sample_id,
                                        **{k:scores[k] for k in ('dice','iou','boundary_f1')}))
            summary = dict(dataset=name, count=len(current),
                           **{k:sum(row[k] for row in current)/len(current) for k in ('dice','iou','boundary_f1')})
            rows.extend(current)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation.')
    with (args.output/'per_image.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(dict(
        checkpoint_sha256=checkpoint_hash, split_sha256=sha256(SOFTWARE/'data/kvasir_split.json'),
        torch=torch.__version__, device=args.device, input_size=256, threshold=0.5,
        metric='per-image macro mean; Dice/IoU smooth=1', calibration=False,
        results=summaries), indent=2)+'\n')


if __name__ == '__main__':
    main()
