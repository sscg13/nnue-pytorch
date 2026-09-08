"""Run with python -m unittest discover -s tests -p test_rule50.py."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import data_loader
from ftperm import ft_permute_impl
from model.config import ModelConfig, NNUELightningConfig
from model.model import NNUEModel
from model.nnue import NNUE
from model.modules.rule50 import Rule50Embedding
from model.utils.serialize import NNUEReader, NNUEWriter

FEATURES = "HalfKAv2_hm^"
BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def batch(clocks):
    fens = [f"{BOARD} {h} 42" for h in clocks]
    native = data_loader.get_sparse_batch_from_fens("HalfKAv2_hm", fens, [0]*len(fens), [82]*len(fens), [0]*len(fens))
    try:
        return native.contents.get_tensors("cpu")
    finally:
        data_loader.destroy_sparse_batch(native)


def evaluate(model, data):
    us, them, white, black, _, _, pc, clock = data
    return model(us, them, white, black, pc, rule50=clock)


class Rule50Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loader_preserves_clock_and_owns_memory(self):
        data = batch([0, 1, 63, 64, 99, 100, 150, 255])
        self.assertEqual(data[-1].tolist(), [0, 1, 63, 64, 99, 100, 150, 255])
        # Native storage was freed; overwrite it with another batch.
        batch([37]*8)
        self.assertEqual(data[-1].tolist(), [0, 1, 63, 64, 99, 100, 150, 255])
        self.assertTrue(torch.equal(data[2][0], data[2][-1]))

    def test_independent_rows_clamping_and_gradient(self):
        table = Rule50Embedding(2, 8192, stacks=8)
        with torch.no_grad():
            table.weight[3, 1].fill_(0.1)
            table.weight[3, 2].fill_(-0.2)
            table.weight[3, 100].fill_(0.3)
        clock = torch.tensor([-1, 0, 1, 2, 100, 150])
        out = table(clock, torch.full_like(clock, 3))
        self.assertTrue(torch.equal(out[0], out[1]))
        self.assertTrue(torch.equal(out[-1], out[-2]))
        self.assertGreater(out[2, 0], 0)
        self.assertLess(out[3, 0], 0)
        out.sum().backward()
        self.assertEqual(table.weight.grad[3, 100, 0].item(), 2)
        self.assertEqual(table.weight.grad[2].count_nonzero().item(), 0)

    def test_variants_zero_init_roundtrip_permutation_and_optimizer(self):
        data = batch([0, 1, 63, 64, 99, 100, 150])
        torch.manual_seed(15)
        base = NNUEModel(FEATURES, ModelConfig(L1=64))
        with torch.no_grad():
            expected = evaluate(base, data)
        hashes = {NNUEWriter.fc_hash(base)}
        for mode in ("ft", "hidden1", "hidden2"):
            with self.subTest(mode=mode):
                config = ModelConfig(L1=64, rule50=mode)
                wrapped = NNUE(NNUELightningConfig(features=FEATURES, model_config=config))
                model = wrapped.model
                result = model.load_state_dict(base.state_dict(), strict=False)
                self.assertEqual(len(result.missing_keys), 1)
                self.assertEqual(result.unexpected_keys, [])
                torch.testing.assert_close(evaluate(model, data), expected, rtol=0, atol=0)
                table = model.input.rule50 if mode == "ft" else model.layer_stacks.rule50
                opts, _ = wrapped.configure_optimizers()
                ids = [id(p) for g in opts[0].param_groups for p in g["params"]]
                self.assertEqual(ids.count(id(table.weight)), 1)
                with torch.no_grad():
                    table.weight.uniform_(-0.2, 0.2)
                value = evaluate(model, data)
                value.sum().backward()
                self.assertGreater(table.weight.grad.count_nonzero().item(), 0)
                before = table.weight.detach().clone()
                opts[0].step()
                self.assertFalse(torch.equal(before, table.weight))
                with torch.no_grad():
                    value = evaluate(model, data)
                    self.assertFalse(torch.equal(value, expected))
                    # Pair-preserving neuron permutation must include the FT clock table.
                    ft_permute_impl(model, np.arange(31, -1, -1))
                    torch.testing.assert_close(evaluate(model, data), value, rtol=0, atol=1e-6)
                for compression in ("none", "leb128"):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "test.nnue"
                        path.write_bytes(NNUEWriter(model, ft_compression=compression, verbose=False).buf)
                        with path.open("rb") as f:
                            loaded = NNUEReader(f, FEATURES, config).model
                            self.assertEqual(f.read(), b"")
                        torch.testing.assert_close(evaluate(loaded, data), value, rtol=0, atol=1e-6)
                        with path.open("rb") as f, self.assertRaises(ValueError):
                            NNUEReader(f, FEATURES, ModelConfig(L1=64))
                hashes.add(NNUEWriter.fc_hash(model))
        self.assertEqual(len(hashes), 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_ft_cuda_parity(self):
        from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS
        if not _HAS_CUPY_KERNELS:
            self.skipTest("CuPy kernels unavailable")
        model = NNUEModel(FEATURES, ModelConfig(L1=64, rule50="ft")).cuda()
        with torch.no_grad():
            model.input.rule50.weight.uniform_(-0.2, 0.2)
        us, them, w, b, _, _, pc, clock = [v.cuda() for v in batch([0, 1, 63, 100])]
        psqt, _ = model.calculate_buckets(pc)
        outputs, grads = [], []
        for backend in ("torch", "fused"):
            model.zero_grad()
            values = model.input(us, them, w, b, psqt, False, True, backend=backend, rule50=clock)
            sum(x.sum() for x in values).backward()
            outputs.append(values[0].detach())
            grads.append(model.input.rule50.weight.grad.detach().clone())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
