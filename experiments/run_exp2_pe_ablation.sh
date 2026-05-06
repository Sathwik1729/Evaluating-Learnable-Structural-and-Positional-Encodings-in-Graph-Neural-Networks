#!/bin/bash
# Experiment 2: PE Ablation on Cora
# Compares: No PE (GCN) vs Random PE (LSPE-rand) vs Laplacian PE (LSPE)
# Person 3 runs this.

set -e

echo "=== Exp 2: PE Ablation ==="

# No PE — standard GCN
python main.py --config experiments/configs/cora_gcn.yaml

# Laplacian PE — LSPE (already done in run_baselines.sh)
python main.py --config experiments/configs/cora_lspe.yaml

# Random PE — use main.py with --pe_mode random flag
# NOTE: You need to add --pe_mode random support to datasets/loaders.py
# OR manually edit cora_lspe.yaml and add pe_mode: random, then handle in main.py
# Simplest: copy cora_lspe.yaml to cora_lspe_randpe.yaml and add pe_mode: random
python main.py --config experiments/configs/cora_lspe_randpe.yaml

echo "Done. Check experiments/results/summary.csv"
