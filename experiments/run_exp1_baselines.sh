#!/bin/bash
# Experiment 1: Full baseline comparison
# Runs GCN/GraphSAGE/GAT/LSPE on Cora, Citeseer, and PubMed.
# Person 3 runs this after confirming all models train.

set -e

CONFIGS=experiments/configs

echo "=== Exp 1: Full Baseline Comparison ==="

echo "--- GCN ---"
python main.py --config "$CONFIGS/cora_gcn.yaml"
python main.py --config "$CONFIGS/citeseer_gcn.yaml"
python main.py --config "$CONFIGS/pubmed_gcn.yaml"

echo "--- GraphSAGE ---"
python main.py --config "$CONFIGS/cora_graphsage.yaml"
python main.py --config "$CONFIGS/citeseer_graphsage.yaml"
python main.py --config "$CONFIGS/pubmed_graphsage.yaml"

echo "--- GAT ---"
python main.py --config "$CONFIGS/cora_gat.yaml"
python main.py --config "$CONFIGS/citeseer_gat.yaml"
python main.py --config "$CONFIGS/pubmed_gat.yaml"

echo "--- LSPE ---"
python main.py --config "$CONFIGS/cora_lspe.yaml"
python main.py --config "$CONFIGS/citeseer_lspe.yaml"
python main.py --config "$CONFIGS/pubmed_lspe.yaml"

echo "Done. Check experiments/results/summary.csv"
