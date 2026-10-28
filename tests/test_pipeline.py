import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import io
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from bank_marketing.data import load_dataset, synthetic_frame, validate_frame
from bank_marketing.train import Trial, GRID
from bank_marketing.experiment import run_experiment


class PipelineTests(unittest.TestCase):
    def test_original_archive_loader_keeps_unknown(self):
        frame = synthetic_frame()
        raw = frame.to_csv(sep=';', index=False).encode()
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, 'w') as archive:
            archive.writestr('bank-full.csv', raw)
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, 'w') as archive:
            archive.writestr('bank.zip', inner.getvalue())
        with patch('urllib.request.urlopen', return_value=io.BytesIO(outer.getvalue())):
            x, y, meta = load_dataset()
        self.assertEqual(len(y), 200)
        self.assertEqual(meta['actual_nan_counts']['job'], 0)
        self.assertEqual(meta['unknown_counts']['job'], int((frame.job == 'unknown').sum()))
        self.assertIn('day', x)
        self.assertNotIn('day_of_week', x)
        self.assertEqual(meta['day_semantics'], 'last contact day of month in bank-full.csv')
        self.assertTrue(meta['raw_csv_sha256'])

    def test_final_evaluation_only_after_selection(self):
        events = []
        x, y, meta = validate_frame(synthetic_frame(500))
        chosen = Trial(GRID[27], best_epoch=1, best_val_loss=.4, model=object())
        chosen.validation_metrics = {'pr_auc': .3}
        def search(train, val, **kwargs):
            self.assertEqual(len(train.y), 320)
            self.assertEqual(len(val.y), 80)
            events.append('search_finished')
            return chosen, [{'status': 'completed'}] * 81
        def prediction(model, data, device):
            self.assertEqual(events, ['search_finished'])
            self.assertIs(model, chosen.model)
            self.assertEqual(len(data.y), 100)
            events.append('test_evaluated')
            return np.full(len(data.y), .2), 0.
        with tempfile.TemporaryDirectory() as directory:
            with patch('bank_marketing.experiment.load_dataset', return_value=(x, y, meta)), \
                 patch('bank_marketing.experiment.search_hyperparameters', side_effect=search), \
                 patch('bank_marketing.experiment.predict', side_effect=prediction) as predict_mock, \
                 patch('bank_marketing.experiment.create_figures'), \
                 patch('bank_marketing.experiment.torch.save'):
                result = run_experiment(output_dir=directory, device='cpu')
                self.assertEqual(predict_mock.call_count, 1)
                self.assertEqual(events, ['search_finished', 'test_evaluated'])
                self.assertEqual(result['feature_mode'], 'realistic_pre_contact')
                parsed = json.loads((Path(directory) / 'metrics.json').read_text())
                self.assertEqual(parsed['search_summary']['total'], 81)

    def test_primary_cannot_search_duration_mode(self):
        with self.assertRaisesRegex(ValueError, 'Primary search'):
            run_experiment(feature_mode='duration_inclusive')

