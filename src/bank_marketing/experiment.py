import json
import platform
from pathlib import Path
import numpy as np
import sklearn
import torch
from .data import load_dataset, split_indices, build_preprocessor, transform_split
from .train import search_hyperparameters, train_configuration, class_weight
from .evaluate import predict, metrics, create_figures
from .baselines import fit_baselines, evaluate_baselines

LIMITATIONS = [
    'Single dataset; single random split; single primary random seed.',
    'Tabular MLP is not necessarily superior to classical ML.',
    'Random IID splitting does not simulate every temporal deployment scenario.',
    'Historical association does not imply causal marketing effect.',
    'Pre-contact mode removes duration; other fields still require availability checks in a concrete deployment.',
    'PR-AUC is sklearn average precision, not trapezoidal PR area.',
]


def run_experiment(*, seed=42, feature_mode='realistic_pre_contact', output_dir='results', device='auto', csv_path=None, sensitivity=False, workers=1):
    if feature_mode != 'realistic_pre_contact':
        raise ValueError('Primary search requires realistic_pre_contact; use --sensitivity for duration_inclusive')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists() or (output / 'search_results.csv').exists():
        raise FileExistsError('Use a fresh output directory to preserve experiment evidence')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    x, y, dataset_meta = load_dataset(csv_path)
    indices = split_indices(y, seed)
    preprocessor = build_preprocessor(x.iloc[indices['train']], feature_mode)
    train_data = transform_split(preprocessor, x, y, indices['train'])
    validation_data = transform_split(preprocessor, x, y, indices['validation'])
    # Test labels are used only for stratification; test features are not transformed during search.
    best, rows = search_hyperparameters(train_data, validation_data, seed=seed, device=device,
                                       output_csv=output / 'search_results.csv', workers=workers)
    baseline_models = fit_baselines(train_data, seed)
    # Selection is now frozen. Exactly one prediction pass on primary MLP test data.
    test_data = transform_split(preprocessor, x, y, indices['test'])
    probabilities, _ = predict(best.model, test_data, device)
    split_meta = {name: {'rows': len(idx), 'positive': int(y[idx].sum()),
                        'negative': int((y[idx] == 0).sum()), 'positive_ratio': float(y[idx].mean())}
                  for name, idx in indices.items()}
    result = {'dataset': dataset_meta, 'split': split_meta, 'feature_mode': feature_mode,
              'duration': 'excluded', 'input_dimension': train_data.x.shape[1],
              'preprocessing': {'fit_on': 'train only', 'numeric': 'StandardScaler',
                                'categorical': 'OneHotEncoder(handle_unknown=ignore)',
                                'feature_names': preprocessor.get_feature_names_out().tolist()},
              'class_distribution': split_meta, 'pos_weight': class_weight(train_data.y),
              'best_hyperparameters': best.params, 'best_epoch': best.best_epoch,
              'selection_metric': 'highest validation average precision; tie: lowest best validation loss, then trial_id',
              'threshold': .5, 'threshold_rule': 'probability > threshold', 'positive_class': 'yes',
              'checkpoint_rule': 'minimum validation weighted BCE; restored before metrics',
              'validation_metrics': best.validation_metrics, 'test_metrics': metrics(test_data.y, probabilities),
              'test_usage': 'one final primary MLP evaluation after all 81 trials; no test-driven changes',
              'search_summary': {'total': len(rows), 'successful': sum(not r['status'].startswith('failed') for r in rows),
                                 'failed': sum(r['status'].startswith('failed') for r in rows),
                                 'completed': sum(r['status'] == 'completed' for r in rows),
                                 'early_stopped': sum(r['status'] == 'early_stopped' for r in rows)},
              **evaluate_baselines(baseline_models, test_data),
              'reproducibility': {'seed': seed, 'python': platform.python_version(), 'pytorch': torch.__version__,
                                  'scikit_learn': sklearn.__version__, 'numpy': np.__version__, 'device': device,
                                  'torch_threads': torch.get_num_threads(), 'search_workers': workers, 'deterministic_algorithms': 'warn_only',
                                  'split_indices_artifact': 'split_indices.json'},
              'limitations': LIMITATIONS}
    def save():
        (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    save()
    (output / 'split_indices.json').write_text(json.dumps({k: v.tolist() for k, v in indices.items()}), encoding='utf-8')
    (output / 'selected_history.json').write_text(json.dumps(best.history, indent=2, allow_nan=False), encoding='utf-8')
    torch.save(best.best_state, output / 'selected_checkpoint.pt')
    create_figures(output / 'figures', best.history, best.best_epoch, test_data.y, probabilities)
    if sensitivity:
        prep = build_preprocessor(x.iloc[indices['train']], 'duration_inclusive')
        train_s = transform_split(prep, x, y, indices['train'])
        val_s = transform_split(prep, x, y, indices['validation'])
        secondary = train_configuration(train_s, val_s, best.params, seed=seed, device=device)
        section = {'feature_mode': 'duration_inclusive', 'purpose': 'feature-availability sensitivity only; not pre-contact deployable performance',
                   'same_split_indices': True, 'seed': seed, 'hyperparameters': best.params,
                   'input_dimension': train_s.x.shape[1], 'status': secondary.status, 'best_epoch': secondary.best_epoch}
        if secondary.model is not None:
            test_s = transform_split(prep, x, y, indices['test'])
            p, _ = predict(secondary.model, test_s, device)
            section.update(validation_metrics=secondary.validation_metrics, test_metrics=metrics(test_s.y, p))
        else:
            section['failure_reason'] = secondary.failure_reason
        result['duration_inclusive_sensitivity'] = section
        save()
    return result

