import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inspect
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bank_marketing.data import ArrayData
from bank_marketing.train import train_configuration, class_weight, GRID, search_hyperparameters, Trial
from bank_marketing.evaluate import predict


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        rng = np.random.default_rng(9)
        self.train = ArrayData(rng.normal(size=(65, 4)).astype('float32'), np.array([1] * 13 + [0] * 52))
        self.val = ArrayData(rng.normal(size=(20, 4)).astype('float32'), np.array([1, 0] * 10))
        self.params = GRID[27]

    def test_train_only_class_weight(self):
        self.assertEqual(class_weight(self.train.y), {'positive': 13, 'negative': 52, 'pos_weight': 4.})
        self.assertNotEqual(class_weight(self.train.y), class_weight(self.val.y))

    def test_tiny_training_and_singleton_tail(self):
        trial = train_configuration(self.train, self.val, self.params, max_epochs=2)
        self.assertEqual(trial.status, 'completed')
        self.assertIsNotNone(trial.model)
        self.assertEqual(trial.history[0]['learning_rate'], self.params['learning_rate'])
        self.assertTrue(np.isfinite(trial.validation_metrics['pr_auc']))

    def test_early_stop_restores_best_state(self):
        losses = iter([1., .5, .7, .8, .5])
        saved_states = []
        def controlled(model, data, device, criterion):
            saved_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            p, _ = predict(model, data, device, criterion)
            return p, next(losses)
        with patch('bank_marketing.train.predict', side_effect=controlled):
            trial = train_configuration(self.train, self.val, self.params, max_epochs=10, patience=2)
        self.assertEqual(trial.status, 'early_stopped')
        self.assertEqual(trial.best_epoch, 2)
        self.assertEqual(trial.best_val_loss, .5)
        for key, value in trial.model.state_dict().items():
            self.assertTrue(torch.equal(value, saved_states[1][key]))
            self.assertTrue(torch.equal(value, trial.best_state[key]))
        self.assertTrue(any(not torch.equal(saved_states[1][k], saved_states[3][k]) for k in saved_states[1]))

    def test_nonfinite_loss_rejected(self):
        self.train.x[:] = np.nan
        trial = train_configuration(self.train, self.val, self.params, max_epochs=1)
        self.assertEqual(trial.status, 'failed_nonfinite')
        self.assertIsNone(trial.model)
        self.assertTrue(trial.failure_reason)

    def test_nonfinite_gradient_rejected(self):
        with patch('torch.nn.utils.clip_grad_norm_', return_value=torch.tensor(float('inf'))):
            trial = train_configuration(self.train, self.val, self.params, max_epochs=1)
        self.assertEqual(trial.status, 'failed_nonfinite')
        self.assertIn('gradient', trial.failure_reason)

    def test_search_signature_and_grid(self):
        self.assertEqual(len(GRID), 81)
        self.assertEqual(len({tuple(p.values()) for p in GRID}), 81)
        self.assertFalse(any('test' in p for p in inspect.signature(search_hyperparameters).parameters))

    def test_search_isolation_selection_and_failed_exclusion(self):
        calls = []
        def fake(train_data, validation_data, params, **kwargs):
            self.assertIs(train_data, self.train)
            self.assertIs(validation_data, self.val)
            calls.append(params)
            trial = Trial(params)
            if len(calls) == 1:
                trial.status = 'failed_nonfinite'
                trial.failure_reason = 'synthetic failure'
                return trial
            trial.model = object()
            trial.best_val_loss = .2 if len(calls) == 4 else .3
            trial.validation_metrics = {'pr_auc': .8 if len(calls) in [3, 4, 5] else .1,
                                        'accuracy': .99 if len(calls) == 2 else .5}
            return trial
        with patch('bank_marketing.train.train_configuration', side_effect=fake), patch('builtins.print'):
            best, rows = search_hyperparameters(self.train, self.val)
        self.assertEqual(len(calls), 81)
        self.assertEqual(best.params, GRID[3])
        self.assertTrue(all(not any('test' in key for key in row) for row in rows))

    def test_repeated_training_same_seed(self):
        a = train_configuration(self.train, self.val, self.params, max_epochs=2)
        b = train_configuration(self.train, self.val, self.params, max_epochs=2)
        self.assertEqual(a.history, b.history)
        self.assertEqual(a.validation_metrics, b.validation_metrics)

