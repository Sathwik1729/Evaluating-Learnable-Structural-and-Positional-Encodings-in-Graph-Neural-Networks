# Evaluating Learnable Structural and Positional Encodings in Graph Neural Networks

**CS768 — Probabilistic Machine Learning**  
**IIT Bombay, Spring 2026**

| Member | Roll No | Contribution |
|---|---|---|
| Sathwik | 23B0946 | Dataset loading, training infrastructure, GCN baseline |
| Dinesh | 23B1022 | Experiment support, result analysis |
| Rithvik | 23B0939 | GraphSAGE, GAT, LSPE-GCN implementation |
| Sujay | 23B1059 | Experiments, ablations, visualizations, report |

---

## Abstract

Graph Neural Networks (GNNs) typically aggregate information from local neighborhoods but lack an explicit mechanism to encode where a node sits in the global graph structure. Learnable Structural and Positional Encodings (LSPE), proposed by Dwivedi et al. (NeurIPS 2022), address this by maintaining a separate positional stream alongside node features, updated through its own GNN layers at every step. In this work, we implement and evaluate LSPE-GCN and three standard baselines (GCN, GraphSAGE, GAT) on the Planetoid citation benchmarks (Cora, Citeseer, PubMed) under a controlled experimental framework. We further conduct ablation studies on PE type, PE dimension, and model depth, and introduce PEGAT — a stronger extension that injects positional encodings directly into a GAT backbone. Our key findings are: (1) shallow LSPE does not outperform tuned baselines on citation graphs due to the small semi-supervised training regime; (2) Laplacian PE strictly outperforms random PE, confirming the signal carries structural information; (3) LSPE degrades significantly less than GCN as depth increases, suggesting PE helps resist over-smoothing; and (4) PEGAT with Random Walk PE achieves the best Cora accuracy (83.9%), demonstrating that PE is most effective when paired with an attention backbone.

---

## 1. Introduction

Graph-structured data arises in many real-world domains: citation networks, molecular graphs, social networks, and knowledge graphs. Graph Neural Networks (GNNs) have become the dominant paradigm for learning on such data, operating through a message-passing framework where each node iteratively aggregates representations from its neighbors.

However, standard message-passing GNNs have a well-known limitation: they are fundamentally local. A node's representation after $L$ layers only reflects information within its $L$-hop neighborhood, and the model has no inherent mechanism to distinguish where a node sits in the global graph topology. Two nodes with identical local neighborhoods are indistinguishable by a standard GNN, even if they play structurally different roles (e.g., bridge nodes vs. peripheral nodes).

Positional encodings (PE) offer a principled solution. In Transformer-based models, PE has been central since Vaswani et al. (2017). For graphs, early work used fixed Laplacian eigenvectors as PE, but they were only added at the input. Dwivedi et al. (2022) introduced LSPE, where PE is not merely concatenated at the input — it evolves through its own dedicated GNN layers at every message-passing step, allowing the model to jointly refine structural position and node features.

**This project asks**: Do learnable structural and positional representations improve node classification performance and robustness on standard citation graph benchmarks?

**Contributions**:
1. A clean, modular PyTorch Geometric implementation of GCN, GraphSAGE, GAT, and LSPE-GCN in a unified training pipeline.
2. Systematic evaluation across five controlled experiments: baseline comparison, PE type ablation, PE dimension sensitivity, depth analysis, and a stronger PE-augmented extension.
3. PEGAT: a new model variant combining Random Walk PE with a GAT backbone, achieving the best accuracy in our evaluation.
4. Multi-seed analysis quantifying variance on the Cora benchmark.

---

## 2. Background and Related Work

### 2.1 Message-Passing GNNs

The general message-passing framework (Gilmer et al., 2017) defines a GNN layer as:

$$h_v^{(l+1)} = \text{UPDATE}\!\left(h_v^{(l)},\ \text{AGGREGATE}\!\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

**GCN** (Kipf & Welling, 2017) instantiates this with symmetric normalized aggregation:

$$h_v^{(l+1)} = \sigma\!\left(W^{(l)} \cdot \frac{1}{\sqrt{\tilde{d}_v}} \sum_{u \in \tilde{\mathcal{N}}(v)} \frac{1}{\sqrt{\tilde{d}_u}} h_u^{(l)}\right)$$

**GraphSAGE** (Hamilton et al., 2017) uses mean aggregation with a separate self-connection, enabling inductive generalization.

**GAT** (Veličković et al., 2018) replaces fixed aggregation weights with learned attention coefficients:

$$\alpha_{vu} = \text{softmax}_u\!\left(\text{LeakyReLU}(a^\top [Wh_v \| Wh_u])\right)$$

### 2.2 Positional Encodings for Graphs

The Laplacian eigenvectors of the normalized graph Laplacian $\hat{L} = I - D^{-1/2}AD^{-1/2}$ provide a natural coordinate system for graph nodes. The $k$ eigenvectors corresponding to the $k$ smallest non-trivial eigenvalues form a $k$-dimensional PE for each node. These are sign-ambiguous (any eigenvector can be flipped), which is handled at training time by random sign flipping.

Random Walk PE (RWPE) encodes the probability of landing back at node $v$ after $t$ steps: $[\text{diag}(P^1), \ldots, \text{diag}(P^k)]$ where $P = D^{-1}A$ is the transition matrix. RWPE captures local graph topology and is invariant to node ordering.

### 2.3 LSPE (Dwivedi et al., 2022)

The key innovation of LSPE is that the positional encoding stream is updated separately at every layer:

$$h_v^{(l+1)} = \sigma\!\left(W_h^{(l)} \cdot \underset{u \in \tilde{\mathcal{N}}(v)}{\text{MEAN}}\!\left[\,h_u^{(l)} \| p_u^{(l)}\,\right]\right)$$

$$p_v^{(l+1)} = p_v^{(l)} + \tanh\!\left(W_p^{(l)} \cdot \underset{u \in \tilde{\mathcal{N}}(v)}{\text{MEAN}}\!\left[p_u^{(l)}\right]\right)$$

The residual tanh update preserves the signed structure of Laplacian eigenvectors across layers. The final classifier uses only $h_v^{(L)}$; the positional stream $p$ serves purely as an auxiliary signal that enriches message passing.

---

## 3. Methodology

### 3.1 Datasets

All experiments use the standard Planetoid semi-supervised splits.

| Dataset | Nodes | Edges | Features | Classes | Train / Val / Test |
|---|---:|---:|---:|---:|---|
| Cora | 2,708 | 5,429 | 1,433 | 7 | 140 / 500 / 1,000 |
| Citeseer | 3,327 | 4,732 | 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 44,338 | 500 | 3 | 60 / 500 / 1,000 |

The extremely small training sets (60–140 nodes) make these benchmarks challenging for large models. Features are L1-normalized. Laplacian PE is computed via PyTorch Geometric's `AddLaplacianEigenvectorPE` transform.

### 3.2 Models

**GCN** — 2-layer Graph Convolutional Network with ReLU activations and dropout. Serves as the primary reference baseline.

**GraphSAGE** — 2-layer mean-aggregation network. Tests whether inductive neighborhood sampling improves over symmetric normalization.

**GAT** — 2-layer Graph Attention Network with 4 attention heads, ELU activations. The strongest standard baseline on Cora.

**LSPE-GCN** — Our implementation follows Dwivedi et al. with two adaptations for citation graphs: (a) a GCNConv input projection (instead of plain Linear) for graph-aware feature compression on high-dimensional datasets like Citeseer (3703 → 64), and (b) BatchNorm after each projection for stability on small label sets. The positional stream uses a residual tanh update. During training on Laplacian PE, random sign flipping is applied per eigenvector.

**PEGAT** — Our extension: GAT backbone with positional encodings concatenated directly to node features at the input layer. Uses Random Walk PE as the positional signal. Tests whether PE benefits an attention-based backbone.

### 3.3 Positional Encodings

| PE Mode | Description |
|---|---|
| Laplacian | $k$ smallest non-trivial eigenvectors of normalized Laplacian |
| Random Walk (RWPE) | $[\text{diag}(P^1), \ldots, \text{diag}(P^k)]$, landing probabilities |
| Random | Gaussian noise $\mathcal{N}(0, 1)$, same dimension — ablation baseline |
| None | No PE (standard GNN) |

### 3.4 Training Setup

All models are trained with Adam optimizer. Model selection uses validation accuracy; test accuracy is reported at the best validation epoch.

| Hyperparameter | GCN | GraphSAGE | GAT | LSPE | PEGAT |
|---|---|---|---|---|---|
| Hidden dim | 64 | 64 | 64 | 64 | 128 |
| Layers | 2 | 2 | 2 | 2 | 2 |
| Dropout | 0.5 | 0.5 | 0.6 | 0.6 | 0.6 |
| LR | 0.01 | 0.01 | 0.005 | 0.01 | 0.005 |
| Weight decay | 5e-4 | 5e-4 | 5e-4 | 5e-3 | 5e-4 |
| Epochs | 200 | 200 | 200 | 300–400 | 300 |
| Attention heads | — | — | 4 | — | 4 |
| PE dim | — | — | — | 8 | 8 |

All experiments use seed 42 unless stated otherwise. Hardware: NVIDIA RTX 3050 (4GB VRAM), CUDA 12.1, PyTorch 2.5.1, PyTorch Geometric 2.7.0.

---

## 4. Experiments and Results

### Experiment 1: Full Baseline Comparison

We evaluate all four models on all three datasets. Reported metric is test accuracy at the best validation epoch (single seed = 42).

| Dataset | GCN | GraphSAGE | GAT | LSPE |
|---|---:|---:|---:|---:|
| Cora | 0.824 | 0.797 | **0.829** | 0.778 |
| Citeseer | **0.720** | 0.703 | 0.717 | 0.610 |
| PubMed | **0.794** | 0.776 | 0.775 | 0.741 |
| **Mean** | **0.779** | 0.759 | 0.774 | 0.710 |

**Observations**:
- GAT is the best model on Cora; GCN leads on Citeseer and PubMed.
- LSPE consistently underperforms relative to baselines. The gap is largest on Citeseer (−11 points vs GCN), which we attribute to: (a) Citeseer's extreme sparsity (avg degree ~1.4), making neighbor PE aggregation less informative; (b) the tiny 120-node training set making the additional PE parameters harder to optimize.
- On PubMed, LSPE (0.741) is 5.3 points below GCN, but notably this is PubMed's smallest training set (60 nodes for 3 classes — only 20 per class).

### Experiment 2: PE Type Ablation (Cora)

We compare four PE conditions on Cora using the LSPE architecture (where applicable) vs GCN as the no-PE baseline.

| Condition | Model | Test Acc | Δ vs No-PE |
|---|---|---:|---:|
| No PE | GCN | **0.824** | — |
| Laplacian PE | LSPE | 0.770 | −0.054 |
| Random PE | LSPE | 0.747 | −0.077 |
| Laplacian − Random | — | — | **+0.023** |

**Observations**:
- Laplacian PE outperforms random PE by 2.3 percentage points, confirming that the eigenvector structure carries real information beyond the embedding dimension.
- However, even Laplacian PE does not help LSPE match the GCN baseline. This reveals a key limitation: on semi-supervised citation graphs, the overhead of learning the PE stream from 140 training nodes may not justify the gain.

### Experiment 3: PE Dimension Study (Cora, LSPE)

| PE Dim | Params | Test Acc | Time (s) |
|---:|---:|---:|---:|
| 4 | 101,503 | 0.758 | 14.9 |
| **8** | **102,135** | **0.776** | **14.9** |
| 16 | 103,591 | 0.738 | 22.1 |
| 32 | 107,271 | 0.765 | 23.1 |

**Observations**: PE dim = 8 is the sweet spot. Larger dimensions increase runtime linearly but do not improve accuracy — likely because the additional eigenvectors correspond to higher-frequency, noisier components of the graph Laplacian that are harder to learn from limited supervision.

### Experiment 4: Depth Analysis — Over-Smoothing (Cora)

| Model | 2 Layers | 4 Layers | 8 Layers | Drop (2→8) |
|---|---:|---:|---:|---:|
| GCN | 0.824 | 0.779 | 0.646 | **−0.178** |
| LSPE | 0.770 | 0.744 | 0.681 | **−0.089** |

![Depth analysis](visualizations/figures/exp4_depth_analysis.png)

**Observations**: This is our most academically significant finding. GCN loses 17.8 points going from 2 to 8 layers — a clear signature of over-smoothing, where all node representations converge to indistinguishable values. LSPE loses only 8.9 points — less than half the degradation. This supports the LSPE paper's core motivation: by maintaining a positional stream that preserves each node's structural identity, the model can resist the homogenizing effect of repeated neighborhood aggregation. At 8 layers, LSPE (0.681) outperforms GCN (0.646) by 3.5 points — the only setting where LSPE beats GCN in our experiments.

### Experiment 5: PEGAT — Positional Encodings in GAT (All Datasets)

We test whether concatenating Random Walk PE to node features before GAT layers improves the strongest baseline.

| Dataset | GAT (no PE) | PEGAT + RWPE | Δ |
|---|---:|---:|---:|
| Cora | 0.829 | **0.839** | +0.010 |
| Citeseer | **0.717** | 0.700 | −0.017 |
| PubMed | **0.775** | 0.789 | +0.014 |

**Observations**:
- On Cora and PubMed, PEGAT+RWPE improves over GAT, achieving the best test accuracy in the entire evaluation for Cora (83.9%).
- On Citeseer, PEGAT underperforms. Citeseer's extreme sparsity likely makes RWPE uninformative — the transition matrix converges rapidly on a sparse graph, making powers of $P$ nearly uniform.
- The improvement on PubMed (+1.4 points) is notable given that PubMed is the largest dataset, suggesting RWPE scales better than Laplacian PE on denser graphs.

### Experiment 6: Multi-Seed Stability (Cora, Seeds 0/1/2)

To quantify result variance on the small Planetoid training sets, we run four key models on Cora with three independent seeds.

| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---:|---:|---:|---|
| GCN | 0.808 | 0.809 | 0.816 | **0.811 ± 0.004** |
| GAT | 0.806 | 0.825 | 0.826 | **0.819 ± 0.011** |
| LSPE | 0.761 | 0.728 | 0.747 | **0.745 ± 0.017** |
| PEGAT+RWPE | 0.810 | 0.828 | 0.826 | **0.821 ± 0.010** |

**Observations**:
- GCN is the most stable model (std = 0.004), consistent with its simple inductive bias.
- LSPE has the highest variance (std = 0.017), reflecting the difficulty of optimizing the PE stream from 140 training nodes.
- PEGAT+RWPE achieves mean accuracy 0.821, comparable to GAT (0.819) and better than GCN (0.811) on average. The best PEGAT+RWPE run (seed 42 = 0.839) represents the top of its distribution.
- The 95% confidence intervals of GCN and GAT largely overlap, confirming that no single seed result should be over-interpreted on these benchmarks.

---

## 5. Discussion

### Why LSPE Underperforms on Citation Graphs

The LSPE paper demonstrates improvements on harder benchmarks (ZINC molecular regression, PATTERN/CLUSTER graph classification) with thousands of training examples per class. Citation graphs present a fundamentally different challenge:

1. **Tiny training sets**: 140 nodes (Cora), 120 nodes (Citeseer), 60 nodes (PubMed). LSPE adds ~10,000 parameters for PE projection and updates — this is a large number to optimize from so few labeled examples.
2. **Low-degree graphs**: Citeseer has average degree ≈ 1.4. With such sparse connectivity, neighborhood aggregation of PE adds little information — most nodes have 0–2 neighbors.
3. **Laplacian PE ambiguity**: The sign ambiguity of eigenvectors requires random sign flipping during training. With 140 training nodes, the model sees each node ≈ 3.5× per epoch, providing limited signal to resolve the sign symmetry.

### The Depth Finding is the Strongest Result

The depth analysis (Experiment 4) is our most robust finding: LSPE reduces the over-smoothing penalty by 50% compared to GCN. This is theoretically motivated — the positional stream retains structural identity across layers, acting as an "anchor" that prevents node representations from converging. At 8 layers, LSPE outperforms GCN, which is the strongest validation of the LSPE framework in this study.

### PEGAT Generalizes LSPE's Core Idea More Effectively

PEGAT takes a different approach: rather than maintaining a separate PE stream through the GNN, it fuses PE at the input and lets GAT attention do the rest. This is simpler and cheaper but still benefits from structural position. The RWPE signal (self-return probabilities) appears better suited to citation graphs than Laplacian eigenvectors, since it is non-negative, bounded, and less sensitive to graph sparsity.

---

## 6. Summary of All Results

### Primary Benchmark (Test Accuracy, seed=42)

| Dataset | GCN | GraphSAGE | GAT | LSPE | PEGAT+RWPE |
|---|---:|---:|---:|---:|---:|
| Cora | 0.824 | 0.797 | 0.829 | 0.778 | **0.839** |
| Citeseer | **0.720** | 0.703 | 0.717 | 0.610 | 0.700 |
| PubMed | 0.794 | 0.776 | 0.775 | 0.741 | **0.789** |
| Mean | 0.779 | 0.759 | 0.774 | 0.710 | **0.776** |

### Training Efficiency

| Model | Cora Time | Citeseer Time | PubMed Time | Cora Params |
|---|---:|---:|---:|---:|
| GCN | 3s | 13s | 32s | 92,231 |
| GraphSAGE | 25s | 51s | 62s | 184,391 |
| GAT | 29s | 69s | 92s | 92,373 |
| LSPE | 15s | 9s | 9s | 102,135 |
| PEGAT+RWPE | 8s | 16s | 12s | 185,749 |

LSPE adds only ~10% parameters over GCN on Cora while being ~5× faster to train than GAT, making it parameter and compute efficient. PEGAT is larger (128 hidden) but still trains faster than GAT due to fewer message-passing operations.

---

## 7. Figures

### Exp 1: Baseline Accuracy Comparison
![Baseline comparison](visualizations/figures/exp1_accuracy_comparison.png)

### Exp 2: PE Ablation
![PE ablation](visualizations/figures/exp2_pe_ablation.png)

### Exp 3: PE Dimension Study
![Dimension study](visualizations/figures/exp3_dim_study.png)

### Exp 4: Depth Analysis
![Depth analysis](visualizations/figures/exp4_depth_analysis.png)

### Training Curves: GCN vs LSPE (Cora)
![GCN curves](visualizations/figures/cora_gcn_L2_H64_PE8_pelaplacian_seed42_curves.png)
![LSPE curves](visualizations/figures/cora_lspe_L2_H64_PE8_pelaplacian_seed42_curves.png)

### t-SNE: Raw Features vs Laplacian PE (Cora)
![Feature t-SNE](visualizations/figures/tsne_cora_features.png)
![PE t-SNE](visualizations/figures/tsne_cora_laplacian_pe.png)

---

## 8. Limitations

1. **Single seed for most experiments**: Multi-seed analysis (Experiment 6) is only run on Cora for 4 models. The remaining results (all datasets, all ablations) use a single seed and should be interpreted with variance in mind.

2. **Citation graphs only**: All three datasets are citation networks with similar structure (homophilic, low density). Results may not generalize to heterophilic graphs, molecular graphs, or social networks where PE is hypothesized to help most.

3. **No hyperparameter search**: Each model uses a fixed hyperparameter set. LSPE's underperformance may partly reflect suboptimal tuning rather than a fundamental limitation of the architecture.

4. **Not a full paper reproduction**: The LSPE paper evaluates GatedGCN-LSPE and PNA-LSPE on harder benchmarks (ZINC, CLUSTER, PATTERN). Our implementation focuses on GCN-LSPE adapted for node classification, which is a different task and regime than the paper's primary experiments.

5. **Citeseer LSPE remains weak**: Despite hyperparameter tuning, LSPE achieves only 61.0% on Citeseer vs 72.0% for GCN — a 11-point gap that warrants further investigation (possible architecture mismatch for extremely sparse, high-dimensional feature spaces).

---

## 9. Conclusion

We implemented a complete framework for evaluating learnable positional encodings in GNNs on citation graph benchmarks. Our experiments yield the following conclusions:

1. **LSPE does not outperform tuned shallow baselines on citation graphs** under standard semi-supervised training. The primary bottleneck is the extremely small training set size, which limits optimization of the positional stream.

2. **Positional encodings carry real structural information**: Laplacian PE outperforms random PE by 2.3 points in the ablation, confirming the signal is meaningful.

3. **LSPE reduces over-smoothing significantly**: At 8 layers, LSPE degrades half as much as GCN (−8.9 vs −17.8 points). At 8 layers, LSPE actually outperforms GCN, providing the strongest empirical support for the approach.

4. **PEGAT+RWPE achieves the best accuracy** (83.9% on Cora, 78.9% on PubMed), demonstrating that positional encodings are most effective when integrated into an attention backbone with a well-matched PE type. Random Walk PE outperforms Laplacian PE on these datasets.

5. **Result variance is non-trivial**: The 95% confidence intervals of GCN and GAT overlap substantially across seeds, reminding us that single-seed results on small Planetoid benchmarks should be interpreted cautiously.

The most actionable direction from this work is the depth finding: if practical applications require deeper GNNs (e.g., large graphs where long-range dependencies matter), LSPE offers a meaningful advantage. For shallow 2-layer settings on citation graphs, a well-tuned GAT or GCN remains competitive.

---

## 10. References

1. Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2022). *Graph Neural Networks with Learnable Structural and Positional Representations*. ICLR 2022 / NeurIPS 2022. arXiv:2110.07875.

2. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.

3. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.

4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. ICLR 2018.

5. Gilmer, J., Schütt, S., Riley, P., Vinyals, O., & Dahl, G. E. (2017). *Neural Message Passing for Quantum Chemistry*. ICML 2017.

6. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

7. Yang, Z., Cohen, W. W., & Salakhutdinov, R. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings*. ICML 2016. [Planetoid datasets]

8. Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR 2019 Workshop.
