import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import torch
from bank_marketing.model import BankMarketingMLP


class ModelTests(unittest.TestCase):
    def test_dimensions_and_finite_logits(self):
        for dim in [3, 17, 50]:
            model = BankMarketingMLP(dim)
            logits = model(torch.randn(4, dim))
            self.assertEqual(tuple(logits.shape), (4, 1))
            self.assertTrue(torch.isfinite(logits).all())

    def test_architecture_and_dropout(self):
        model = BankMarketingMLP(10, .3, .4)
        self.assertEqual([m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)], [.3, .4])
        self.assertEqual([m.out_features for m in model.modules() if isinstance(m, torch.nn.Linear)], [128, 64, 1])
        self.assertEqual([m.negative_slope for m in model.modules() if isinstance(m, torch.nn.LeakyReLU)], [.01, .01])

    def test_initialization_reproducible(self):
        from bank_marketing.train import seed_everything
        seed_everything(42)
        first = BankMarketingMLP(10)
        seed_everything(42)
        second = BankMarketingMLP(10)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

