"""
TODO (Person 4): Implement all visualization functions.

Run this file directly to generate all plots:
  python visualizations/plots.py

Plots are saved to visualizations/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = 'experiments/results/summary.csv'
RESULTS_DIR = 'experiments/results'
FIGURES_DIR = 'visualizations/figures'

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    return pd.read_csv(RESULTS_CSV)


def plot_accuracy_comparison(df=None):
    """
    TODO: Bar chart comparing test accuracy of all 4 models on all 3 datasets.

    Group bars by dataset (Cora, Citeseer, PubMed).
    Each group has 4 bars: GCN, GraphSAGE, GAT, LSPE.
    Highlight LSPE bar in a different color.

    Save to: visualizations/figures/exp1_accuracy_comparison.png
    """
    raise NotImplementedError("Person 4: implement plot_accuracy_comparison()")


def plot_training_curves(run_name):
    """
    TODO: Line plot of train_loss and val_acc vs epoch for a given run.

    Load from: experiments/results/{run_name}_history.npy
    Save to:   visualizations/figures/{run_name}_curves.png

    Example run_name: 'cora_gcn_L2_H64_PE8_seed42'
    """
    raise NotImplementedError("Person 4: implement plot_training_curves()")


def plot_pe_ablation(df=None):
    """
    TODO: Bar chart for Experiment 2 — PE ablation on Cora.
    X-axis: No PE, Random PE, Laplacian PE
    Y-axis: Test accuracy

    Save to: visualizations/figures/exp2_pe_ablation.png
    """
    raise NotImplementedError("Person 4: implement plot_pe_ablation()")


def plot_dim_study(df=None):
    """
    TODO: Line plot for Experiment 3 — pe_dim sensitivity.
    X-axis: pe_dim values (4, 8, 16, 32)
    Y-axis: Test accuracy (LSPE on Cora)

    Save to: visualizations/figures/exp3_dim_study.png
    """
    raise NotImplementedError("Person 4: implement plot_dim_study()")


def plot_depth_analysis(df=None):
    """
    TODO: Line plot for Experiment 4 — depth analysis.
    X-axis: num_layers (2, 4, 8)
    Y-axis: Test accuracy
    Two lines: GCN vs LSPE (on Cora)

    Shows whether LSPE mitigates over-smoothing.
    Save to: visualizations/figures/exp4_depth_analysis.png
    """
    raise NotImplementedError("Person 4: implement plot_depth_analysis()")


if __name__ == '__main__':
    df = load_results()
    print(df.to_string())

    plot_accuracy_comparison(df)
    plot_pe_ablation(df)
    plot_dim_study(df)
    plot_depth_analysis(df)

    # Plot training curves for key runs
    key_runs = [
        'cora_gcn_L2_H64_PE8_seed42',
        'cora_lspe_L2_H64_PE8_seed42',
    ]
    for run in key_runs:
        path = os.path.join(RESULTS_DIR, f'{run}_history.npy')
        if os.path.exists(path):
            plot_training_curves(run)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
