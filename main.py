"""
Entry point for all experiments.

Usage:
  python main.py --config experiments/configs/cora_gcn.yaml
  python main.py --config experiments/configs/cora_gcn.yaml --epochs 400 --seed 0
"""
import argparse
import yaml
import torch

from datasets import load_dataset
from models import build_model
from training import train, set_seed, get_device, save_results
from training.utils import count_parameters, save_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    # Allow CLI overrides of any config key
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--pe_dim', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--heads', type=int)
    return parser.parse_args()


def load_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Apply CLI overrides
    for key in ['dataset', 'model', 'hidden_dim', 'num_layers', 'dropout',
                'lr', 'weight_decay', 'epochs', 'pe_dim', 'seed', 'heads']:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    return config


def main():
    args = parse_args()
    config = load_config(args)

    print("\n" + "="*50)
    print(f"Dataset : {config['dataset'].upper()}")
    print(f"Model   : {config['model'].upper()}")
    print(f"Layers  : {config['num_layers']}  Hidden: {config['hidden_dim']}")
    print(f"PE dim  : {config.get('pe_dim', 8)}  Seed: {config.get('seed', 42)}")
    print("="*50 + "\n")

    set_seed(config.get('seed', 42))
    device = get_device()
    print(f"Device: {device}\n")

    # Load dataset
    data, num_features, num_classes = load_dataset(
        name=config['dataset'],
        pe_dim=config.get('pe_dim', 8),
    )
    print(f"Nodes: {data.num_nodes}  Edges: {data.edge_index.size(1)}  "
          f"Features: {num_features}  Classes: {num_classes}\n")

    # Build model
    model = build_model(config, num_features, num_classes)
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}\n")

    # Train
    best_val, best_test, history, train_time = train(
        model, data, config, device, verbose=True
    )

    # Build run name for saving
    run_name = (f"{config['dataset']}_{config['model']}"
                f"_L{config['num_layers']}_H{config['hidden_dim']}"
                f"_PE{config.get('pe_dim',8)}_seed{config.get('seed',42)}")

    # Save results
    results = {
        'dataset': config['dataset'],
        'model': config['model'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
        'lr': config['lr'],
        'pe_dim': config.get('pe_dim', 8),
        'seed': config.get('seed', 42),
        'val_acc': round(best_val, 4),
        'test_acc': round(best_test, 4),
        'train_time_s': round(train_time, 1),
        'num_params': num_params,
    }
    save_results(results)
    save_history(
        {'train_loss': history['train_loss'],
         'val_acc': history['val_acc'],
         'test_acc': history['test_acc']},
        run_name=run_name,
    )

    print(f"\nRun: {run_name}")
    print(f"Val Acc:  {best_val:.4f}")
    print(f"Test Acc: {best_test:.4f}")


if __name__ == '__main__':
    main()
