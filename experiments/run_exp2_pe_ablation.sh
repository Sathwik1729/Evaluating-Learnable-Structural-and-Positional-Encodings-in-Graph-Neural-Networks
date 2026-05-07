#!/bin/bash
# Experiment 2: PE Ablation on Cora
# Compares: No PE (GCN) vs Random PE (LSPE-rand) vs Laplacian PE (LSPE)
# Person 3 runs this after Person 2 has implemented all models.

set -e

echo "=== Exp 2: PE Ablation ==="

# Condition 1: No PE — standard GCN (no positional signal at all)
echo "--- No PE: GCN ---"
python main.py --config experiments/configs/cora_gcn.yaml

# Condition 2: Laplacian PE — LSPE with meaningful structural positions
echo "--- Laplacian PE: LSPE ---"
python main.py --config experiments/configs/cora_lspe.yaml

# Condition 3: Random PE — LSPE with random noise instead of eigenvectors
# Tests whether the PE content matters or just its existence
echo "--- Random PE: LSPE-randpe ---"
python main.py --config experiments/configs/cora_lspe_randpe.yaml

echo "Done. Check experiments/results/summary.csv"
