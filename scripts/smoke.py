"""Offline synthetic smoke training; never portfolio evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from bank_marketing.data import synthetic_frame, validate_frame, split_indices, build_preprocessor, transform_split
from bank_marketing.train import train_configuration, GRID


def main():
    torch.set_num_threads(1)
    x, y, _ = validate_frame(synthetic_frame())
    idx = split_indices(y)
    prep = build_preprocessor(x.iloc[idx['train']])
    train = transform_split(prep, x, y, idx['train'])
    val = transform_split(prep, x, y, idx['validation'])
    trial = train_configuration(train, val, GRID[27], max_epochs=2)
    assert trial.model is not None, trial.failure_reason
    print('Synthetic smoke passed:', trial.status, trial.best_epoch)


if __name__ == '__main__':
    main()

