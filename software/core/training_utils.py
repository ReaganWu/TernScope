"""Shared optimizer and precision conversion from the supplied training source."""
import torch
import topology as r
CFG = {"lr": 2e-4, "weight_decay": 1e-5}

def seed():
 r.seed(42);torch.cuda.manual_seed_all(42)


def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'],foreach=False)


