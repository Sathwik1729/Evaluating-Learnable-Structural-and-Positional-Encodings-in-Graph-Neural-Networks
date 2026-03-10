# Evaluating Learnable Structural and Positional Encodings in Graph Neural Networks

## Project Proposal

Graph Neural Networks (GNNs) are widely used for learning from graph-structured data such as citation networks, social networks, and biological interaction networks. Most GNN models rely on message passing between neighboring nodes to compute node representations. However, these models often lack explicit mechanisms to encode structural or positional information about nodes within a graph.

In this project, we study the **Learnable Structural and Positional Encoding (LSPE)** framework for graph neural networks. LSPE augments node representations with learnable structural and positional embeddings that provide additional information about the role and position of nodes in the graph.

Our goal is to implement LSPE-based models and evaluate how these encodings influence node classification performance compared to standard GNN architectures.

---

## Team Members

- Your Name (Roll Number)
- Teammate Name (Roll Number)

---

## Objectives

- Understand the limitations of standard message passing GNNs
- Study the role of structural and positional information in graph learning
- Implement baseline GNN models (GCN / GraphSAGE)
- Implement the LSPE architecture
- Evaluate the impact of positional encodings on node classification performance

---

## Datasets

We plan to evaluate our models on standard citation network datasets commonly used in graph learning research:

- **Cora**
- **Citeseer**
- **PubMed**

These datasets are widely used benchmarks for node classification tasks.

---

## Experiments

The project will include the following experiments:

1. Implementation of baseline GNN models
2. Implementation of LSPE-based GNN models
3. Comparison of node classification accuracy
4. Ablation studies analyzing the effect of positional encodings

---

## Main Reference

Dwivedi, V. P. et al.  
**Graph Neural Networks with Learnable Structural and Positional Representations**  
NeurIPS 2022  

Paper link:  
https://arxiv.org/abs/2110.07875