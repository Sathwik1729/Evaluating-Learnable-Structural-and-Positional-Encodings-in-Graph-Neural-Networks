#!/bin/bash
# Experiment 3: Embedding dimension study — pe_dim in {4, 8, 16, 32}
# Person 3 runs this after LSPE is working.

set -e
echo "=== Exp 3: PE Dimension Study (Cora, LSPE) ==="

for dim in 4 8 16 32; do
    echo "--- pe_dim=$dim ---"
    python main.py --config experiments/configs/cora_lspe.yaml --pe_dim $dim
done

echo "Done."
