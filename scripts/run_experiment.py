"""Run from repository root: python -m scripts.run_experiment --sensitivity."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from bank_marketing.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature-mode', choices=['realistic_pre_contact'], default='realistic_pre_contact',
                        help='Primary mode. Use --sensitivity for the duration-inclusive comparison.')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--csv-path', help='Optional local original UCI CSV')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--workers', type=int, default=1, help='Independent CPU search processes')
    parser.add_argument('--threads', type=int, default=1)
    args = vars(parser.parse_args())
    torch.set_num_threads(args.pop('threads'))
    result = run_experiment(**args)
    print('Primary MLP experiment completed:', result['test_metrics'], flush=True)


if __name__ == '__main__':
    main()

