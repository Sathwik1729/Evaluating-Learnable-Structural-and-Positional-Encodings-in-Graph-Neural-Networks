#!/bin/bash
# Run all baseline models on all datasets (Person 2 runs this after implementing GraphSAGE/GAT)

set -e
CONFIGS=experiments/configs

echo "=== GCN ==="
python main.py --config $CONFIGS/cora_gcn.yaml
python main.py --config $CONFIGS/citeseer_gcn.yaml
python main.py --config $CONFIGS/pubmed_gcn.yaml

echo "=== GraphSAGE ==="
python main.py --config $CONFIGS/cora_graphsage.yaml
python main.py --config $CONFIGS/citeseer_graphsage.yaml
python main.py --config $CONFIGS/pubmed_graphsage.yaml

echo "=== GAT ==="
python main.py --config $CONFIGS/cora_gat.yaml
python main.py --config $CONFIGS/citeseer_gat.yaml
python main.py --config $CONFIGS/pubmed_gat.yaml

echo "=== LSPE ==="
python main.py --config $CONFIGS/cora_lspe.yaml
python main.py --config $CONFIGS/citeseer_lspe.yaml
python main.py --config $CONFIGS/pubmed_lspe.yaml

echo "Done. Results in experiments/results/summary.csv"
