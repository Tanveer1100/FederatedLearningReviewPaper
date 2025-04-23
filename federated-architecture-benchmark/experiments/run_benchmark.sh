#!/bin/bash

# Architectures to test
architectures=("centralized" "decentralized" "hierarchical")

# Datasets to test (as described in the paper)
datasets=("FEMNIST" "Sent140" "Shakespeare" "CIFAR10" "Synthetic")

# Optional: Number of clients per round (you can tweak this)
clients_per_round=10

# Optional: Total number of training rounds
rounds=50

# Loop through all combinations
for arch in "${architectures[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "=========================================================="
    echo "Running Federated Learning Benchmark"
    echo "Architecture: $arch | Dataset: $dataset"
    echo "=========================================================="
    python trainer/train_fl_architectures.py \
      --dataset "$dataset" \
      --architecture "$arch" \
      --clients_per_round "$clients_per_round" \
      --rounds "$rounds"
  done
done
