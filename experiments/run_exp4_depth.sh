#!/bin/bash
# Experiment 4: Depth analysis — num_layers in {2, 4, 8}
# Tests over-smoothing behavior in GCN vs LSPE.
# Person 3 runs this.

set -e
echo "=== Exp 4: Depth Analysis (Cora) ==="

for layers in 2 4 8; do
    echo "--- GCN, layers=$layers ---"
    python main.py --config experiments/configs/cora_gcn.yaml --num_layers $layers

    echo "--- LSPE, layers=$layers ---"
    python main.py --config experiments/configs/cora_lspe.yaml --num_layers $layers
done

echo "Done."
