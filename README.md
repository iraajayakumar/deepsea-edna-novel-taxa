# Identifying Novel Taxonomy from eDNA Datasets using Multi-scale FCGR and Invariant Information Clustering

This repository contains the full implementation and experimental code for the paper:

**“Identifying Novel Taxonomy from eDNA Datasets using Multi-scale FCGR and Invariant Information Clustering”**

The work presents an end-to-end, taxonomy-agnostic deep learning pipeline for unsupervised clustering and novel taxon discovery from environmental DNA (eDNA) metabarcoding data, with a focus on deep-sea marine sediments.

---

## Overview

Environmental DNA datasets are highly fragmented, long-tailed, and dominated by rare or unknown taxa, making reference-based identification unreliable.  
This project proposes a **self-supervised clustering framework** that:

- Avoids sequence alignment and reference databases
- Preserves rare and low-abundance biological signals
- Identifies putative novel taxa through clustering instability and ensemble disagreement

The pipeline integrates **multi-scale FCGR encoding**, **abundance-aware mimic generation**, **Invariant Information Clustering (IIC)**, and **ensemble consensus analysis** into a unified workflow.

---

## Method Summary

The core components of the pipeline are:

- **Multi-k Frequency Chaos Game Representation (FCGR)**  
  Alignment-free sequence encoding at k = 4, 5, 6 to capture compositional patterns across scales.

- **Abundance-Aware Mimic Generation**  
  Self-supervised augmentation strategy that applies tiered perturbations based on sequence abundance to protect rare taxa.

- **Multi-Branch CNN with Attention Fusion**  
  Independent FCGR branches fused using learnable attention for adaptive multiscale feature integration.

- **Invariant Information Clustering (IIC)**  
  Mutual information maximization between original–mimic pairs with relaxed entropy constraints to handle long-tailed abundance distributions.

- **Ensemble Clustering and Alignment**  
  Multiple independently trained models aligned via the Hungarian algorithm and aggregated for stability.

- **Novelty Assessment**  
  Candidate novel taxa identified using confidence, entropy, ensemble disagreement, and cluster size criteria.


