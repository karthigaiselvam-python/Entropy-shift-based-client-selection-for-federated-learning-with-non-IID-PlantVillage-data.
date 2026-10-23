# Entropy-Shift-Based Client Selection in Federated Learning

This repository contains the Python implementation of five federated learning experiments investigating entropy-shift-based client selection with Non-IID agricultural image data.

## Project Overview

The experiments use the PlantVillage dataset for plant disease classification. A CNN is trained using federated learning with 20 simulated clients: 10 IID clients and 10 Non-IID clients. Non-IID data is generated using a Dirichlet distribution with α = 0.1.

Each experiment runs for 100 communication rounds using FedAvg.

## Experiments

1. **FedAvg Baseline:** All 20 clients participate without entropy-based selection.
2. **Median + MAD:** Client updates are selected using a threshold calculated from all clients' predictive entropy shifts.
3. **Adaptive EMA:** An adaptive threshold is used to select client updates based on entropy shifts.
4. **Non-IID-Only Median + MAD:** All IID clients are included, while entropy-based selection is applied to Non-IID clients.
5. **Decaying Client-Side Threshold:** Clients decide whether to upload their model weights using a progressively decreasing entropy-shift threshold.

## Dataset

PlantVillage dataset, containing 38 plant disease and healthy leaf classes.

The dataset is not included in this repository.

## Evaluation

Each experiment evaluates global model classification accuracy across 100 communication rounds.

The experiments investigate whether entropy-shift-based selection can maintain competitive accuracy while reducing the number of client updates used or uploaded.
