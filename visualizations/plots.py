"""
Generate plots for the LSPE experiment suite.

Run from the repository root:
  python visualizations/plots.py

Plots are saved to visualizations/figures/.
"""
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('visualizations/.matplotlib')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_CSV = Path('experiments/results/summary.csv')
RESULTS_DIR = Path('experiments/results')
FIGURES_DIR = Path('visualizations/figures')

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = ['cora', 'citeseer', 'pubmed']
MODEL_ORDER = ['gcn', 'graphsage', 'gat', 'lspe']
MODEL_LABELS = {
    'gcn': 'GCN',
    'graphsage': 'GraphSAGE',
    'gat': 'GAT',
    'lspe': 'LSPE',
}
DATASET_LABELS = {
    'cora': 'Cora',
    'citeseer': 'Citeseer',
    'pubmed': 'PubMed',
}
COLORS = {
    'gcn': '#4C78A8',
    'graphsage': '#F58518',
    'gat': '#54A24B',
    'lspe': '#B279A2',
    'random': '#E45756',
    'laplacian': '#72B7B2',
    'none': '#9D755D',
}


def _format_axes(ax, title, xlabel='', ylabel='Test accuracy'):
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def _to_numeric(df):
    numeric_cols = [
        'hidden_dim', 'num_layers', 'dropout', 'lr', 'pe_dim', 'seed',
        'val_acc', 'test_acc', 'train_time_s', 'num_params', 'csv_row',
    ]
    for col in numeric_cols:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().all():
                df[col] = converted
    return df


def load_results():
    """
    Load summary.csv while handling both historical row formats:
    old rows do not contain pe_mode, newer rows do.
    """
    rows = []
    old_header = [
        'dataset', 'model', 'hidden_dim', 'num_layers', 'dropout', 'lr',
        'pe_dim', 'seed', 'val_acc', 'test_acc', 'train_time_s', 'num_params',
    ]
    new_header = [
        'dataset', 'model', 'hidden_dim', 'num_layers', 'dropout', 'lr',
        'pe_dim', 'pe_mode', 'seed', 'val_acc', 'test_acc',
        'train_time_s', 'num_params',
    ]

    with RESULTS_CSV.open(newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for csv_row, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) == len(new_header):
                item = dict(zip(new_header, row))
            elif len(row) == len(old_header):
                item = dict(zip(old_header, row))
                item['pe_mode'] = 'laplacian'
            else:
                raise ValueError(
                    f"Unexpected number of columns in {RESULTS_CSV}:{csv_row}: {len(row)}"
                )
            item['csv_row'] = csv_row
            rows.append(item)

    return _to_numeric(pd.DataFrame(rows))


def _rows(df, start, end):
    return df[(df['csv_row'] >= start) & (df['csv_row'] <= end)].copy()


def _last_matching(df, **criteria):
    subset = df.copy()
    for key, value in criteria.items():
        subset = subset[subset[key] == value]
    if subset.empty:
        raise ValueError(f"No result row found for {criteria}")
    return subset.sort_values('csv_row').iloc[-1]


def _annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
        )


def plot_accuracy_comparison(df=None):
    """
    Bar chart comparing test accuracy of all 4 models on all 3 datasets.
    Saves to visualizations/figures/exp1_accuracy_comparison.png.
    """
    df = load_results() if df is None else df
    exp1 = _rows(df, 43, 54)
    if exp1.empty:
        exp1 = pd.DataFrame([
            _last_matching(
                df, dataset=dataset, model=model, num_layers=2,
                pe_dim=8, pe_mode='laplacian',
            )
            for model in MODEL_ORDER
            for dataset in DATASET_ORDER
        ])

    x = np.arange(len(DATASET_ORDER))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, model in enumerate(MODEL_ORDER):
        values = []
        for dataset in DATASET_ORDER:
            row = exp1[(exp1['dataset'] == dataset) & (exp1['model'] == model)]
            values.append(float(row.iloc[0]['test_acc']) if not row.empty else np.nan)
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
        _annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylim(0.50, 0.88)
    _format_axes(ax, 'Experiment 1: Baseline Test Accuracy by Dataset')
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.10))
    _savefig(FIGURES_DIR / 'exp1_accuracy_comparison.png')


def plot_training_curves(run_name):
    """
    Plot train loss, validation accuracy, and test accuracy for a saved run.
    Saves to visualizations/figures/{run_name}_curves.png.
    """
    path = RESULTS_DIR / f'{run_name}_history.npy'
    history = np.load(path, allow_pickle=True).item()
    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_acc = ax_loss.twinx()

    loss_line, = ax_loss.plot(
        epochs, history['train_loss'], color='#4C78A8', lw=2, label='Train loss'
    )
    val_line, = ax_acc.plot(
        epochs, history['val_acc'], color='#54A24B', lw=2, label='Val acc'
    )
    test_line, = ax_acc.plot(
        epochs, history['test_acc'], color='#E45756', lw=2, label='Test acc'
    )

    ax_loss.set_title(f'Training Curves: {run_name}', fontsize=12, weight='bold')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_acc.set_ylabel('Accuracy')
    ax_loss.grid(alpha=0.25)
    ax_loss.spines['top'].set_visible(False)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.set_ylim(0.0, 0.9)

    lines = [loss_line, val_line, test_line]
    ax_loss.legend(lines, [line.get_label() for line in lines], frameon=False)
    _savefig(FIGURES_DIR / f'{run_name}_curves.png')


def plot_pe_ablation(df=None):
    """
    Bar chart for Experiment 2: no PE vs random PE vs Laplacian PE on Cora.
    Saves to visualizations/figures/exp2_pe_ablation.png.
    """
    df = load_results() if df is None else df
    exp2 = _rows(df, 55, 57)
    if exp2.empty:
        exp2 = pd.DataFrame([
            _last_matching(df, dataset='cora', model='gcn', num_layers=2, pe_dim=8),
            _last_matching(df, dataset='cora', model='lspe', pe_mode='random'),
            _last_matching(df, dataset='cora', model='lspe', pe_mode='laplacian'),
        ])

    conditions = [
        ('No PE', exp2[exp2['model'] == 'gcn']),
        ('Random PE', exp2[(exp2['model'] == 'lspe') & (exp2['pe_mode'] == 'random')]),
        ('Laplacian PE', exp2[(exp2['model'] == 'lspe') & (exp2['pe_mode'] == 'laplacian')]),
    ]
    labels = [label for label, _ in conditions]
    values = [float(rows.iloc[0]['test_acc']) for _, rows in conditions]
    colors = [COLORS['none'], COLORS['random'], COLORS['laplacian']]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    _annotate_bars(ax, bars)
    ax.set_ylim(0.70, 0.84)
    _format_axes(ax, 'Experiment 2: Positional Encoding Ablation on Cora')
    _savefig(FIGURES_DIR / 'exp2_pe_ablation.png')


def plot_dim_study(df=None):
    """
    Line plot for Experiment 3: LSPE pe_dim sensitivity on Cora.
    Saves to visualizations/figures/exp3_dim_study.png.
    """
    df = load_results() if df is None else df
    exp3 = _rows(df, 58, 61)
    if exp3.empty:
        exp3 = pd.DataFrame([
            _last_matching(
                df, dataset='cora', model='lspe', num_layers=2,
                pe_dim=dim, pe_mode='laplacian',
            )
            for dim in [4, 8, 16, 32]
        ])
    exp3 = exp3.sort_values('pe_dim')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        exp3['pe_dim'], exp3['test_acc'],
        marker='o', markersize=7, lw=2.5, color=COLORS['lspe'],
    )
    for _, row in exp3.iterrows():
        ax.annotate(
            f"{row['test_acc']:.3f}",
            (row['pe_dim'], row['test_acc']),
            textcoords='offset points',
            xytext=(0, 8),
            ha='center',
            fontsize=9,
        )
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(0.72, 0.79)
    _format_axes(ax, 'Experiment 3: LSPE Sensitivity to PE Dimension', 'PE dimension')
    _savefig(FIGURES_DIR / 'exp3_dim_study.png')


def plot_depth_analysis(df=None):
    """
    Line plot for Experiment 4: GCN vs LSPE depth analysis on Cora.
    Saves to visualizations/figures/exp4_depth_analysis.png.
    """
    df = load_results() if df is None else df
    exp4 = _rows(df, 62, 67)
    if exp4.empty:
        exp4 = pd.DataFrame([
            _last_matching(
                df, dataset='cora', model=model, num_layers=layers,
                pe_dim=8, pe_mode='laplacian',
            )
            for layers in [2, 4, 8]
            for model in ['gcn', 'lspe']
        ])

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ['gcn', 'lspe']:
        rows = exp4[exp4['model'] == model].sort_values('num_layers')
        ax.plot(
            rows['num_layers'], rows['test_acc'],
            marker='o', markersize=7, lw=2.5,
            color=COLORS[model], label=MODEL_LABELS[model],
        )
        for _, row in rows.iterrows():
            ax.annotate(
                f"{row['test_acc']:.3f}",
                (row['num_layers'], row['test_acc']),
                textcoords='offset points',
                xytext=(0, 8),
                ha='center',
                fontsize=9,
            )

    ax.set_xticks([2, 4, 8])
    ax.set_ylim(0.60, 0.85)
    _format_axes(ax, 'Experiment 4: Depth and Over-smoothing on Cora', 'Number of layers')
    ax.legend(frameon=False)
    _savefig(FIGURES_DIR / 'exp4_depth_analysis.png')


if __name__ == '__main__':
    df = load_results()
    print(df.tail(12).to_string(index=False))

    plot_accuracy_comparison(df)
    plot_pe_ablation(df)
    plot_dim_study(df)
    plot_depth_analysis(df)

    key_runs = [
        'cora_gcn_L2_H64_PE8_pelaplacian_seed42',
        'cora_lspe_L2_H64_PE8_pelaplacian_seed42',
        'cora_gat_L2_H64_PE8_pelaplacian_seed42',
    ]
    for run in key_runs:
        path = RESULTS_DIR / f'{run}_history.npy'
        if path.exists():
            plot_training_curves(run)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
