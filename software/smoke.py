"""CPU checks of fusion, ternary coverage, gradients, and frozen checkpoint reload."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from runtime import fixed_split, load_w2a8, quantization as q, topology as r
from ternscope import create_ternscope


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_branch_fusion(self):
        model = create_ternscope(input_size=256).eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            before = model(x)
            fused = copy.deepcopy(model).switch_to_deploy()
            after = fused(x)
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)
        deployment = q.QuantGraph(fused, 'int8')
        deployment.disable()  # Drops the two unused training-only auxiliary heads.
        with torch.inference_mode():
            torch.testing.assert_close(after, deployment(x), rtol=1e-4, atol=1e-4)
        self.assertEqual(len(deployment.coverage), 25)
        self.assertEqual(sum(p.numel() for p in r.TernScopeDeploy().parameters()), 248273)

    def test_ternary_coverage_gradient_and_reload(self):
        model = q.QuantGraph(r.TernScopeDeploy(), 'ternary')
        for module in model.modules():
            if isinstance(module, q.QConv):
                module.bits = 2
        convs = [m for m in model.modules() if isinstance(m, q.QConv)]
        self.assertEqual(len(convs), 25)
        self.assertEqual(len([m for m in model.modules() if isinstance(m, q.Act8)]), 53)
        for module in convs:
            codes = module.quant_weight()/module.log_scale.exp().clamp_min(1e-8)
            torch.testing.assert_close(codes, codes.round(), rtol=0, atol=1e-5)
            self.assertLessEqual(float(codes.detach().abs().max()), 1.00001)
        x = torch.randn(1, 3, 64, 64)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        r.loss_fn(logits, torch.zeros_like(logits)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.pth'
            torch.save({'model':model.state_dict()}, path)
            restored = load_w2a8(path)
            with torch.inference_mode():
                torch.testing.assert_close(restored(x), model.eval()(x), rtol=0, atol=0)
        self.assertEqual([len(fixed_split()[k]) for k in ('train','validation','test')], [720,80,200])


if __name__ == '__main__':
    unittest.main(verbosity=2)
