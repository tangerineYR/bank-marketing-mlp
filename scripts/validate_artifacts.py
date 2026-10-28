"""Validate saved experiment evidence without evaluating any model again."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bank_marketing.train import GRID


def validate(output):
    output = Path(output)
    result = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((output / 'search_results.csv').open(encoding='utf-8', newline='')))
    assert len(rows) == 81
    assert {int(row['trial_id']) for row in rows} == set(range(1, 82))
    assert all('test' not in key.lower() for key in rows[0])
    assert {tuple(float(row[k]) for k in GRID[0]) for row in rows} == {tuple(p.values()) for p in GRID}
    good = [r for r in rows if r['status'] in ('completed', 'early_stopped')]
    selected = min(good, key=lambda r: (-float(r['val_pr_auc']), float(r['best_val_loss']), int(r['trial_id'])))
    for k, value in result['best_hyperparameters'].items():
        assert float(selected[k]) == value
    assert result['best_epoch'] == int(selected['best_epoch'])
    assert result['validation_metrics']['pr_auc'] == float(selected['val_pr_auc'])
    assert result['feature_mode'] == 'realistic_pre_contact'
    assert result['duration'] == 'excluded'
    assert 'numeric__duration' not in result['preprocessing']['feature_names']
    assert 'numeric__day' in result['preprocessing']['feature_names']
    assert 'numeric__day_of_week' not in result['preprocessing']['feature_names']
    assert result['dataset']['day_semantics'] == 'last contact day of month in bank-full.csv'
    assert 'day' in result['dataset']['actual_nan_counts']
    assert 'day_of_week' not in result['dataset']['actual_nan_counts']
    assert result['input_dimension'] == len(result['preprocessing']['feature_names'])
    def finite(value):
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
        elif isinstance(value, (int, float)):
            assert math.isfinite(value)
    finite(result)
    for row in good:
        for key, value in row.items():
            if key.startswith('val_') or key == 'best_val_loss':
                assert math.isfinite(float(value))
    for row in rows:
        if row['status'].startswith('failed'):
            assert row['failure_reason']
            assert not row['val_pr_auc']
    splits = json.loads((output / 'split_indices.json').read_text())
    flattened = [i for values in splits.values() for i in values]
    assert len(flattened) == len(set(flattened)) == result['dataset']['total_rows']
    for key, values in splits.items():
        assert len(values) == result['split'][key]['rows']
    for key in ['test_metrics', 'dummy_baseline', 'logistic_regression_baseline']:
        assert sum(result[key]['confusion_matrix'].values()) == len(splits['test'])
    history = json.loads((output / 'selected_history.json').read_text())
    assert min(history, key=lambda h: h['val_loss'])['epoch'] == result['best_epoch']
    for name in ['training_curve', 'confusion_matrix', 'precision_recall_curve', 'roc_curve']:
        assert (output / 'figures' / (name + '.png')).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'historical' / 'provenance.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256((root / 'historical' / 'baseline_2025.py').read_bytes()).hexdigest() == provenance['sha256']
    print('Artifact validation passed: 81 trials, selection, finite metrics, split integrity, figures, historical hash.')


if __name__ == '__main__':
    validate(sys.argv[1] if len(sys.argv) > 1 else 'results')

