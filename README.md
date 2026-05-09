# Evaluating Learnable Structural and Positional Encodings in Graph Neural Networks

This project evaluates whether learnable structural and positional encodings improve node classification in Graph Neural Networks. We compare standard message-passing baselines against an LSPE-GCN model inspired by Dwivedi et al., "Graph Neural Networks with Learnable Structural and Positional Representations" (NeurIPS 2022).

The experiments use the Planetoid citation benchmarks: Cora, Citeseer, and PubMed.

## Team

| Member | Roll No | Responsibility |
|---|---|---|
| Sathwik | 23B0946 | Dataset loading, training infrastructure, GCN baseline |
| Dinesh | 23B1022 | Experiment support |
| Rithvik | 23B0939 | GraphSAGE, GAT, LSPE-GCN implementation |
| Sujay | 23B1059 | Experiments, result tables, visualizations |

## Models

| Model | Description |
|---|---|
| GCN | Graph Convolutional Network baseline |
| GraphSAGE | Mean aggregation baseline |
| GAT | Attention-based graph baseline |
| LSPE | Improved GCN-style model with projected PE, residual tanh positional updates, and random sign flipping for Laplacian PE |
| LSPE Legacy | Original project LSPE kept for old-vs-new comparison |
| PEGAT | Stronger update: GAT backbone with positional encodings concatenated to input features |

LSPE uses positional encodings stored as `data.pe`. The positional stream is updated separately at each layer using a residual tanh update, and the final classifier uses the learned node feature stream only. Supported PE modes: `laplacian`, `rw` (Random Walk), `random`.

## Repository Structure

```text
datasets/                 Dataset loading and PE utilities
models/                   GCN, GraphSAGE, GAT, LSPE, and PEGAT implementations
training/                 Training loop, evaluation, result saving
experiments/configs/      YAML configs for all model/dataset runs
experiments/results/      CSV results, tables, and training histories
visualizations/plots.py   Main plotting script
visualizations/tsne.py    Optional t-SNE visualizations
visualizations/figures/   Generated figures
main.py                   Experiment entry point
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch Geometric wheel installation needs a CUDA-specific command on your machine, follow the official PyG install selector, then rerun the requirements command.

Verify the setup with a single Cora run:

```bash
python main.py --config experiments/configs/cora_gcn.yaml
```

Results are appended to `experiments/results/summary.csv`, and per-epoch histories are saved as `.npy` files in `experiments/results/`.

## Running Experiments

Run the full baseline sweep:

```bash
bash experiments/run_exp1_baselines.sh
```

Run the PE ablation:

```bash
bash experiments/run_exp2_pe_ablation.sh
```

Run the PE dimension study:

```bash
bash experiments/run_exp3_dim_study.sh
```

Run the depth analysis:

```bash
bash experiments/run_exp4_depth.sh
```

Run the old-vs-improved paper update comparison:

```bash
bash experiments/run_paper_update_comparison.sh
```

This compares GCN, GAT, the original LSPE, improved LSPE variants, and the PEGAT extension on Cora. It writes `experiments/results/update_comparison.md` and `visualizations/figures/update_comparison.png`.

Generate result plots:

```bash
python visualizations/plots.py
```

Generate optional t-SNE plots:

```bash
python visualizations/tsne.py
```

## Results

Full detailed tables are available in `experiments/results/tables.md`. The main baseline comparison is:

| Dataset | GCN | GraphSAGE | GAT | LSPE | PEGAT+RWPE |
|---|---:|---:|---:|---:|---:|
| Cora | 0.824 | 0.797 | 0.829 | 0.778 | **0.839** |
| Citeseer | **0.720** | 0.703 | 0.717 | 0.610 | 0.700 |
| PubMed | 0.794 | 0.776 | 0.775 | 0.741 | **0.789** |
| Mean | 0.779 | 0.759 | 0.774 | 0.710 | **0.776** |

Values are test accuracy at the best validation epoch.

### PE Ablation on Cora

| Condition | Model | Test Acc | Δ vs No-PE |
|---|---|---:|---:|
| No PE | GCN | **0.824** | — |
| Laplacian PE | LSPE | 0.770 | −0.054 |
| Random PE | LSPE | 0.747 | −0.077 |

Laplacian PE improved over random PE by 0.023, confirming that the structure of the positional signal matters. However, the LSPE variant did not outperform the tuned GCN baseline on Cora.

### PE Dimension Study on Cora

| PE Dim | Test Acc |
|---:|---:|
| 4 | 0.758 |
| 8 | **0.776** |
| 16 | 0.738 |
| 32 | 0.765 |

The best LSPE result used `pe_dim=8`. Larger PE dimensions increased runtime and parameters without improving accuracy.

### Depth Analysis on Cora

| Model | 2 Layers | 4 Layers | 8 Layers | Drop from 2 to 8 |
|---|---:|---:|---:|---:|
| GCN | 0.824 | 0.779 | 0.646 | -0.178 |
| LSPE | 0.770 | 0.744 | 0.681 | -0.089 |

GCN was stronger at 2 layers, but LSPE degraded less as depth increased (−0.089 vs −0.178). This supports the motivation for structural and positional encodings as a mechanism to reduce over-smoothing in deeper networks.

### Paper Update: Old vs Improved LSPE and PEGAT Extension (Cora)

After the main experiments, the LSPE model was revised to more closely follow the paper (residual tanh PE update, random sign flipping for Laplacian PE, Random Walk PE support). A stronger PEGAT variant was also added. Results on Cora:

| Method | Val Acc | Test Acc | Params | Time (s) |
|---|---:|---:|---:|---:|
| GCN baseline | 0.804 | 0.820 | 92,231 | 3.3 |
| GAT baseline | 0.812 | 0.810 | 92,373 | 4.8 |
| LSPE Legacy + LapPE | 0.778 | 0.786 | 102,135 | 8.4 |
| Improved LSPE + LapPE | 0.742 | 0.743 | 102,103 | 7.6 |
| Improved LSPE + RWPE | 0.734 | 0.731 | 102,103 | 8.7 |
| PEGAT + LapPE | 0.808 | 0.818 | 92,885 | 6.8 |
| **PEGAT + RWPE** | **0.828** | **0.839** | 185,749 | 7.1 |

**Best result: PEGAT + RWPE at 83.9% test accuracy on Cora**, surpassing all other configurations. Random Walk PE provided a stronger positional signal than Laplacian PE for this architecture.

Run `bash experiments/run_paper_update_comparison.sh` to reproduce this table and generate the comparison figure.

## Figures

### Baseline Accuracy

![Baseline accuracy comparison](visualizations/figures/exp1_accuracy_comparison.png)

### Positional Encoding Ablation

![PE ablation](visualizations/figures/exp2_pe_ablation.png)

### PE Dimension Study

![PE dimension study](visualizations/figures/exp3_dim_study.png)

### Depth Analysis

![Depth analysis](visualizations/figures/exp4_depth_analysis.png)

### Paper Update Comparison

![Update comparison](visualizations/figures/update_comparison.png)

### Training Curves

![GCN training curves](visualizations/figures/cora_gcn_L2_H64_PE8_pelaplacian_seed42_curves.png)

![LSPE training curves](visualizations/figures/cora_lspe_L2_H64_PE8_pelaplacian_seed42_curves.png)

### t-SNE Visualizations

![Cora raw feature t-SNE](visualizations/figures/tsne_cora_features.png)

![Cora Laplacian PE t-SNE](visualizations/figures/tsne_cora_laplacian_pe.png)

## Final Report Summary

### Research Question

Do learnable structural and positional encodings improve node classification compared with standard GNN message passing on citation graphs?

### Method

We implemented GCN, GraphSAGE, GAT, LSPE-GCN, and PEGAT in a shared training pipeline. Each model was evaluated on Cora, Citeseer, and PubMed using validation accuracy for model selection and test accuracy for final reporting. Additional controlled experiments tested PE type, PE dimension, and model depth.

### Findings

GAT achieved the best Cora result among the main baselines, while GCN led on Citeseer and PubMed. In the standard benchmark setting, LSPE did not outperform the strongest shallow baselines. However, three findings support the value of positional information:

1. **Laplacian PE beats random PE** in the ablation (+0.023), showing the structural signal matters.
2. **LSPE degrades less with depth** (−0.089 vs −0.178 for GCN from 2 to 8 layers), suggesting PE helps resist over-smoothing.
3. **PEGAT + RWPE achieves 83.9%** on Cora, the best result in the project, showing that positional information can benefit stronger backbone architectures when the PE type is well-matched.

### Limitations

The Planetoid splits have very small training sets, which makes larger models harder to optimize. Results are reported for a single seed; multiple seeds would be needed for statistical confidence. LSPE introduces extra parameters and runtime, especially on Citeseer where it underperforms significantly.

### Conclusion

Learnable positional encodings did not consistently outperform tuned shallow baselines in the main benchmark, but showed clear benefits in the depth analysis and in the PEGAT extension. The strongest result (PEGAT + RWPE, 83.9%) suggests that positional encodings are most effective when integrated into an already-strong attention backbone, and that PE type selection matters as much as the encoding architecture.

## Reference

Dwivedi, V. P. et al. Graph Neural Networks with Learnable Structural and Positional Representations. NeurIPS 2022.

Paper: https://arxiv.org/abs/2110.07875
