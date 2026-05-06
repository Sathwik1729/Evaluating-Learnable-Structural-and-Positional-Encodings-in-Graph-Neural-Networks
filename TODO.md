# Team Task Breakdown — 2 Day Sprint

## Overview
Stack: Python 3.10, PyTorch 2.x, PyTorch Geometric  
Entry point: `python main.py --config experiments/configs/<name>.yaml`

---

## SATHWIK — Day 1 (Today) ✅ DONE WHEN TAGGED

### Done by you:
- [x] Repo structure + environment setup
- [x] `datasets/loaders.py` — dataset loading with Laplacian PE
- [x] `training/utils.py` — seed, device, checkpointing, results logging
- [x] `training/train.py` — training loop
- [x] `training/evaluate.py` — evaluation metrics
- [x] `models/gcn.py` — full GCN implementation
- [x] `models/__init__.py` — model registry (Person 2 just adds entries)
- [x] `experiments/configs/` — config files for all baseline + LSPE experiments
- [x] `main.py` — entry point
- [x] Stubs for Person 2 in `models/` with clear interfaces

### Verify before handing off:
```bash
conda activate lspe
python main.py --config experiments/configs/cora_gcn.yaml
```
Should train GCN on Cora and print val/test accuracy.

---

## PERSON 2 — Day 2 Morning (pick up from Sathwik)

**First: read `models/gcn.py` and `main.py` to understand the interface.**

### Tasks (in order):
1. `models/graphsage.py` — implement GraphSAGE (30 min, SAGEConv from PyG)
2. `models/gat.py` — implement GAT (30 min, GATConv from PyG)
3. `models/lspe.py` — implement LSPE-GCN (core work, 2-3 hours)
   - See stub in file — equations and architecture are fully documented
   - Uses `data.pe` (Laplacian eigenvectors, already in data object)
   - Key: PE has its own separate GNN update path at every layer
4. Register all three in `models/__init__.py` (instructions inside)
5. Run sanity checks:
   ```bash
   python main.py --config experiments/configs/cora_graphsage.yaml
   python main.py --config experiments/configs/cora_gat.yaml
   python main.py --config experiments/configs/cora_lspe.yaml
   ```
6. Run full baseline sweep if time allows:
   ```bash
   bash experiments/run_baselines.sh
   ```

### LSPE Architecture (from paper — Dwivedi et al. 2022):
Each layer updates two streams in parallel:
```
h_v^{l+1} = ReLU(W_h · MEAN_{u ∈ N(v)∪{v}}( concat(h_u^l, p_u^l) ))
p_v^{l+1} = ReLU(W_p · MEAN_{u ∈ N(v)∪{v}}( p_u^l ))
```
Final classifier uses h^L only (not p). See `models/lspe.py` stub for skeleton.

---

## PERSON 3 — Day 2 Afternoon (needs all models working)

**First: confirm all 4 models train by running `bash experiments/run_baselines.sh`**

### Tasks (in order):
1. Run Experiment 1 — Full baseline comparison:
   ```bash
   bash experiments/run_exp1_baselines.sh
   ```
   Collects results for GCN/GraphSAGE/GAT/LSPE on Cora/Citeseer/PubMed

2. Run Experiment 2 — PE Ablation:
   ```bash
   bash experiments/run_exp2_pe_ablation.sh
   ```
   Compares: No PE (GCN) vs Random PE vs Laplacian PE (LSPE)

3. Run Experiment 3 — Embedding Dimension Study:
   ```bash
   bash experiments/run_exp3_dim_study.sh
   ```
   pe_dim ∈ {4, 8, 16, 32} on Cora with LSPE

4. Run Experiment 4 — Depth Analysis:
   ```bash
   bash experiments/run_exp4_depth.sh
   ```
   num_layers ∈ {2, 4, 8} for GCN and LSPE on Cora

5. Compile all results from `experiments/results/summary.csv` into tables
6. Create `experiments/results/tables.md` with formatted result tables

---

## PERSON 4 — Day 2 Afternoon/Evening (needs results from Person 3)

**First: confirm `experiments/results/summary.csv` has data.**

### Tasks (in order):
1. `visualizations/plots.py` — implement all plotting functions:
   - `plot_training_curves()` — loss/accuracy per epoch
   - `plot_accuracy_comparison()` — bar chart comparing all models
   - `plot_pe_ablation()` — ablation bar chart
   - `plot_dim_study()` — line plot for pe_dim sensitivity
   - `plot_depth_analysis()` — over-smoothing analysis
2. Run: `python visualizations/plots.py` to generate all plots to `visualizations/figures/`
3. (Optional) `visualizations/tsne.py` — t-SNE of node embeddings
4. Update `README.md` with:
   - Setup instructions
   - How to run experiments
   - Results table (from Person 3)
   - Example plots (embed figures)
5. Final report sections — write/compile based on results

---

## Dependency Chain
```
Sathwik (today) → Person 2 (morning) → Person 3 (afternoon) → Person 4 (evening)
                                     └→ Person 4 can start README/report in parallel
```

## Key Files Reference
| File | Owner | Status |
|------|-------|--------|
| `datasets/loaders.py` | Sathwik | ✅ |
| `training/train.py` | Sathwik | ✅ |
| `training/evaluate.py` | Sathwik | ✅ |
| `training/utils.py` | Sathwik | ✅ |
| `models/gcn.py` | Sathwik | ✅ |
| `models/graphsage.py` | Person 2 | 🔲 |
| `models/gat.py` | Person 2 | 🔲 |
| `models/lspe.py` | Person 2 | 🔲 |
| `main.py` | Sathwik | ✅ |
| Experiment scripts | Person 3 | 🔲 |
| `visualizations/plots.py` | Person 4 | 🔲 |
