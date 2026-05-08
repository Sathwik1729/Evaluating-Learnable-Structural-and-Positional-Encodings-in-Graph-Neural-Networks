#!/bin/bash
# Run all baseline models on all datasets (Person 2 runs this after implementing GraphSAGE/GAT)

set -e
CONFIGS=experiments/configs

echo "=== GCN ==="
python3 main.py --config $CONFIGS/cora_gcn.yaml
python3 main.py --config $CONFIGS/citeseer_gcn.yaml
python3 main.py --config $CONFIGS/pubmed_gcn.yaml

echo "=== GraphSAGE ==="
python3 main.py --config $CONFIGS/cora_graphsage.yaml
python3 main.py --config $CONFIGS/citeseer_graphsage.yaml
python3 main.py --config $CONFIGS/pubmed_graphsage.yaml

echo "=== GAT ==="
python3 main.py --config $CONFIGS/cora_gat.yaml
python3 main.py --config $CONFIGS/citeseer_gat.yaml
python3 main.py --config $CONFIGS/pubmed_gat.yaml

echo "=== LSPE ==="
python3 main.py --config $CONFIGS/cora_lspe.yaml
python3 main.py --config $CONFIGS/citeseer_lspe.yaml
python3 main.py --config $CONFIGS/pubmed_lspe.yaml

echo "Done. Results in experiments/results/summary.csv"
