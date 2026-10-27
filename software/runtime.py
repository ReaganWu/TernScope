"""Frozen W2A8 loading and preprocessing; no calibration or training."""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
from PIL import Image
import torch

SOFTWARE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOFTWARE / 'core'))
import topology as topology
import quantization as quantization
from ternscope.training import PolypSegmentationDataset, SegmentationSample, SegmentationMetrics


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fixed_split():
    split = json.loads((SOFTWARE / 'data/kvasir_split.json').read_text())
    groups = [split[k] for k in ('train', 'validation', 'test')]
    if [len(g) for g in groups] != [720, 80, 200] or len(set(sum(groups, []))) != 1000:
        raise ValueError('Expected the fixed disjoint 720/80/200 split.')
    return split


def load_w2a8(path, device='cpu', trusted_checkpoint=False):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=not trusted_checkpoint)
    except pickle.UnpicklingError as error:
        raise ValueError('Legacy training checkpoint: use --trusted-checkpoint only for your own trusted artifact, or export a tensor-only model.') from error
    state = checkpoint.get('model', checkpoint)
    model = quantization.QuantGraph(topology.TernScopeDeploy().eval(), 'ternary')
    for name, module in model.named_modules():
        if isinstance(module, quantization.QConv):
            if checkpoint.get('weight_bits', {}).get(name, 2) != 2:
                raise ValueError(f'{name} is not ternary; use a fully W2A8 checkpoint.')
            module.bits = 2
        if isinstance(module, quantization.Act8):
            for key in ('abits', 'adaptive', 'updates'):
                if name + '.' + key in state:
                    module.register_buffer(key, torch.zeros_like(state[name + '.' + key]))
            if checkpoint.get('activation_bits', {}).get(name, 8) != 8:
                raise ValueError(f'{name} is not A8.')
    model.load_state_dict(state, strict=True)
    for name, module in model.named_modules():
        if isinstance(module, quantization.Act8):
            if hasattr(module, 'abits') and int(module.abits) != 8:
                raise ValueError(f'{name} has an incompatible activation-bit buffer.')
            if not torch.isfinite(module.scale) or module.scale <= 0:
                raise ValueError(f'{name} has an invalid activation scale.')
    for entry in model.coverage:
        entry.update(weight_bits_storage=2, levels=3)
    return model.to(device).eval()


def image_tensor(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0


def configure(device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for inference.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
