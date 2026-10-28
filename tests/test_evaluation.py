import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import numpy as np
from bank_marketing.evaluate import metrics


class EvaluationTests(unittest.TestCase):
    def test_known_metrics(self):
        m = metrics([0, 0, 1, 1], [.1, .8, .4, .9])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['precision'], .5)
        self.assertEqual(m['recall'], .5)
        self.assertEqual(m['f1'], .5)
        self.assertEqual(m['roc_auc'], .75)
        self.assertAlmostEqual(m['pr_auc'], 5 / 6)
        self.assertEqual(m['confusion_matrix'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})

    def test_threshold_boundary(self):
        m = metrics([0, 1], [.5, .5])
        self.assertEqual(m['confusion_matrix'], {'tn': 1, 'fp': 0, 'fn': 1, 'tp': 0})
        self.assertEqual(m['precision'], 0.)

    def test_invalid_probabilities(self):
        for p in [[np.nan, .2], [.2, np.inf], [-.1, .2], [.1, 1.1]]:
            with self.assertRaises(ValueError):
                metrics([0, 1], p)

