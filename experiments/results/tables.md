# Experiment Result Tables

Source: `experiments/results/summary.csv`

Values report best validation accuracy and test accuracy at the best validation epoch. The CSV contains earlier trial runs, so these tables use the final completed experiment blocks:

- Experiment 1: rows 43-54
- Experiment 2: rows 55-57
- Experiment 3: rows 58-61
- Experiment 4: rows 62-67

## Experiment 1: Full Baseline Comparison

| Dataset | Model | Hidden Dim | Layers | PE Dim | PE Mode | Val Acc | Test Acc | Time (s) | Params |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Cora | GCN | 64 | 2 | 8 | laplacian | 0.810 | 0.824 | 7.8 | 92,231 |
| Citeseer | GCN | 64 | 2 | 8 | laplacian | 0.726 | 0.720 | 12.8 | 237,446 |
| PubMed | GCN | 64 | 2 | 8 | laplacian | 0.806 | 0.794 | 31.9 | 32,259 |
| Cora | GraphSAGE | 64 | 2 | 8 | laplacian | 0.790 | 0.797 | 24.6 | 184,391 |
| Citeseer | GraphSAGE | 64 | 2 | 8 | laplacian | 0.712 | 0.703 | 51.2 | 474,822 |
| PubMed | GraphSAGE | 64 | 2 | 8 | laplacian | 0.806 | 0.776 | 61.8 | 64,451 |
| Cora | GAT | 64 | 2 | 8 | laplacian | 0.818 | 0.829 | 29.2 | 92,373 |
| Citeseer | GAT | 64 | 2 | 8 | laplacian | 0.734 | 0.717 | 68.5 | 237,586 |
| PubMed | GAT | 64 | 2 | 8 | laplacian | 0.804 | 0.775 | 92.3 | 32,393 |
| Cora | LSPE | 64 | 2 | 8 | laplacian | 0.776 | 0.778 | 14.8 | 102,135 |
| Citeseer | LSPE | 256 | 2 | 8 | laplacian | 0.626 | 0.582 | 76.3 | 1,087,158 |
| PubMed | LSPE | 64 | 2 | 8 | laplacian | 0.724 | 0.727 | 99.2 | 42,163 |

### Experiment 1 Summary

| Dataset | Best Model | Best Test Acc | Runner-up | Runner-up Test Acc |
|---|---|---:|---|---:|
| Cora | GAT | 0.829 | GCN | 0.824 |
| Citeseer | GCN | 0.720 | GAT | 0.717 |
| PubMed | GCN | 0.794 | GraphSAGE | 0.776 |

| Model | Cora Test | Citeseer Test | PubMed Test | Mean Test Acc |
|---|---:|---:|---:|---:|
| GCN | 0.824 | 0.720 | 0.794 | 0.779 |
| GraphSAGE | 0.797 | 0.703 | 0.776 | 0.759 |
| GAT | 0.829 | 0.717 | 0.775 | 0.774 |
| LSPE | 0.778 | 0.582 | 0.727 | 0.696 |

## Experiment 2: PE Ablation on Cora

| Condition | Model | PE Mode | Val Acc | Test Acc | Time (s) | Params |
|---|---|---|---:|---:|---:|---:|
| No PE | GCN | ignored by model | 0.810 | 0.824 | 7.7 | 92,231 |
| Laplacian PE | LSPE | laplacian | 0.758 | 0.770 | 14.7 | 102,135 |
| Random PE | LSPE | random | 0.738 | 0.747 | 13.6 | 102,135 |

### Experiment 2 Summary

| Comparison | Test Acc Difference |
|---|---:|
| Laplacian PE - Random PE | +0.023 |
| Laplacian PE - No PE | -0.054 |
| Random PE - No PE | -0.077 |

## Experiment 3: PE Dimension Study on Cora

| PE Dim | Model | Val Acc | Test Acc | Time (s) | Params |
|---:|---|---:|---:|---:|---:|
| 4 | LSPE | 0.762 | 0.758 | 14.9 | 101,503 |
| 8 | LSPE | 0.766 | 0.776 | 14.9 | 102,135 |
| 16 | LSPE | 0.738 | 0.738 | 22.1 | 103,591 |
| 32 | LSPE | 0.756 | 0.765 | 23.1 | 107,271 |

### Experiment 3 Summary

| Best PE Dim | Best Test Acc | Observation |
|---:|---:|---|
| 8 | 0.776 | PE dim 8 gave the strongest result in this sweep. Larger PE dimensions increased parameters and runtime without improving accuracy. |

## Experiment 4: Depth Analysis on Cora

| Model | Layers | Val Acc | Test Acc | Time (s) | Params |
|---|---:|---:|---:|---:|---:|
| GCN | 2 | 0.810 | 0.824 | 5.6 | 92,231 |
| LSPE | 2 | 0.766 | 0.770 | 14.8 | 102,135 |
| GCN | 4 | 0.782 | 0.779 | 9.5 | 100,551 |
| LSPE | 4 | 0.752 | 0.744 | 33.1 | 111,911 |
| GCN | 8 | 0.652 | 0.646 | 16.5 | 117,191 |
| LSPE | 8 | 0.696 | 0.681 | 56.1 | 131,463 |

### Experiment 4 Summary

| Model | 2 Layers | 4 Layers | 8 Layers | Drop from 2 to 8 |
|---|---:|---:|---:|---:|
| GCN | 0.824 | 0.779 | 0.646 | -0.178 |
| LSPE | 0.770 | 0.744 | 0.681 | -0.089 |

LSPE starts below GCN at 2 layers but degrades less severely as depth increases, which is consistent with the depth study's over-smoothing motivation.
