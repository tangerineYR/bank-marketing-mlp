"""Search accepts only training and validation arrays; never test data."""
from dataclasses import dataclass, field
from itertools import product
import csv
from concurrent.futures import ProcessPoolExecutor
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .model import BankMarketingMLP
from .evaluate import predict, metrics

GRID = [dict(zip(('learning_rate', 'batch_size', 'dropout1', 'dropout2'), values))
        for values in product([.1, .001, .0005], [32, 64, 128], [.3, .4, .5], [.3, .4, .5])]


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weight(y):
    pos, neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if pos == 0 or neg == 0 or pos + neg != len(y):
        raise ValueError('Training labels must contain both binary classes')
    return {'positive': pos, 'negative': neg, 'pos_weight': neg / pos}


@dataclass
class Trial:
    params: dict
    status: str = 'completed'
    failure_reason: str = ''
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    validation_metrics: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    model: object = None
    best_state: dict = field(default_factory=dict)


def train_configuration(train_data, validation_data, params, *, seed=42, device='cpu', max_epochs=200, patience=10):
    if max_epochs < 1 or patience < 1 or params['batch_size'] < 2 or len(train_data.y) < 2:
        raise ValueError('Invalid training size or settings')
    seed_everything(seed)
    trial = Trial(params.copy())
    model = BankMarketingMLP(train_data.x.shape[1], params['dropout1'], params['dropout2']).to(device)
    weight = class_weight(train_data.y)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    batch_size = min(params['batch_size'], len(train_data.y))
    # Drop only a singleton tail: BatchNorm cannot train on one example.
    drop_last = len(train_data.y) % batch_size == 1
    loader = DataLoader(TensorDataset(torch.from_numpy(train_data.x),
                        torch.tensor(train_data.y, dtype=torch.float32).reshape(-1, 1)),
                        batch_size=batch_size, shuffle=True, drop_last=drop_last,
                        generator=torch.Generator().manual_seed(seed))
    stale = 0
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            total, count = 0., 0
            epoch_lr = optimizer.param_groups[0]['lr']
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5, error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite gradient norm')
                optimizer.step()
                total += loss.item() * len(x)
                count += len(x)
            _, val_loss = predict(model, validation_data, device, criterion)
            trial.history.append({'epoch': epoch, 'train_loss': total / count, 'val_loss': val_loss, 'learning_rate': epoch_lr})
            scheduler.step()
            if val_loss < trial.best_val_loss:
                trial.best_val_loss, trial.best_epoch, stale = val_loss, epoch, 0
                trial.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    trial.status = 'early_stopped'
                    break
        model.load_state_dict(trial.best_state)
        probabilities, _ = predict(model, validation_data, device, criterion)
        trial.validation_metrics = metrics(validation_data.y, probabilities)
        trial.model = model
    except FloatingPointError as exc:
        trial.status, trial.failure_reason = 'failed_nonfinite', str(exc)
        trial.model, trial.best_state, trial.validation_metrics = None, {}, {}
    return trial



def _worker_trial(args):
    # Each process has independent RNG state; one Torch thread prevents oversubscription.
    train_data, validation_data, params, seed, device = args
    torch.set_num_threads(1)
    return train_configuration(train_data, validation_data, params, seed=seed, device=device)


def _trial_iterator(train_data, validation_data, seed, device, workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers == 1:
        for params in GRID:
            yield train_configuration(train_data, validation_data, params, seed=seed, device=device)
    else:
        if device != 'cpu':
            raise ValueError('Parallel workers are supported on CPU only')
        arguments = [(train_data, validation_data, params, seed, device) for params in GRID]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map yields in grid order, preserving deterministic tie breaking and CSV order.
            yield from pool.map(_worker_trial, arguments)


def search_hyperparameters(train_data, validation_data, *, seed=42, device='cpu', output_csv=None, workers=1):
    """All 81 historical configurations. Rank AP, then loss, then grid order."""
    best, best_rank, rows = None, None, []
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    for trial_id, trial in enumerate(_trial_iterator(train_data, validation_data, seed, device, workers), 1):
        params = trial.params
        row = {'trial_id': trial_id, **params, 'seed': seed, 'status': trial.status,
               'failure_reason': trial.failure_reason, 'best_epoch': trial.best_epoch,
               'best_val_loss': trial.best_val_loss if np.isfinite(trial.best_val_loss) else '',
               **{f'val_{key}': trial.validation_metrics.get(key, '') for key in metric_names}}
        rows.append(row)
        if trial.model is not None:
            rank = (-trial.validation_metrics['pr_auc'], trial.best_val_loss, trial_id)
            if best_rank is None or rank < best_rank:
                best, best_rank = trial, rank
        if output_csv:
            with open(output_csv, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
        print(f'Trial {trial_id}/81: {trial.status}; best_epoch={trial.best_epoch}; val_AP={row["val_pr_auc"]}', flush=True)
    if best is None:
        raise RuntimeError('All trials failed; no test evaluation performed')
    return best, rows

