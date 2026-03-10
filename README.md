# Evaluating Structural and Positional Encodings in Graph Neural Networks

## Overview

Graph Neural Networks (GNNs) are widely used for learning from graph-structured data such as citation networks, social networks, and molecular graphs. Most GNN models rely on message passing between neighboring nodes to compute node representations. While effective, these models often rely primarily on local connectivity and may not explicitly capture the structural role or position of nodes within the graph.

This project investigates **Learnable Structural and Positional Encodings (LSPE)** for Graph Neural Networks. The LSPE framework augments node representations with additional learnable embeddings that encode structural and positional information in the graph. These embeddings are learned jointly with the model parameters and integrated into the message-passing process.

The objective of this project is to implement LSPE-based models and evaluate how structural and positional encodings influence node classification performance compared to standard GNN architectures.

---

## Team

- Sathwik – 23B0946
- Dinesh - 23B1022
- Rithvik - 23B0939
- Sujay - 23B1022
---

## Project Goals

The main goals of this project are:

- Study the limitations of standard message-passing GNN architectures
- Understand how structural and positional information can improve node representations
- Implement baseline GNN models for comparison
- Implement the LSPE framework
- Evaluate the impact of positional encodings through controlled experiments

---

## Methodology

The project will proceed in three stages:

1. **Baseline Models**  
   Implement standard GNN architectures such as Graph Convolutional Networks (GCN) to establish baseline performance.

2. **LSPE Implementation**  
   Implement GNN models augmented with learnable structural and positional encodings.

3. **Experimental Evaluation**  
   Compare baseline models and LSPE models in terms of node classification performance and analyze the effect of positional encodings.

---

## Datasets

Experiments will be conducted on commonly used graph learning benchmarks:

- **Cora**
- **Citeseer**
- **PubMed**

These datasets contain citation networks where nodes represent papers and edges represent citation relationships. The task is node classification.

---


---

## Reference

Dwivedi, V. P. et al.  
*Graph Neural Networks with Learnable Structural and Positional Representations*  
NeurIPS 2022  

Paper:  
https://arxiv.org/abs/2110.07875
