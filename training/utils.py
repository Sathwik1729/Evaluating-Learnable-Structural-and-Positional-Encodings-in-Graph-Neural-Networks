import os
import random
import csv
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(results, results_dir='experiments/results'):
    """Append one run's results to summary.csv."""
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'summary.csv')
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Results saved to {csv_path}")


def save_history(history, run_name, results_dir='experiments/results'):
    """Save per-epoch training history as a numpy file."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'{run_name}_history.npy')
    np.save(path, history)
