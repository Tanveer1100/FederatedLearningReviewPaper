#!/bin/bash

# experiments/run_benchmark.sh
# Run full federated learning benchmark as described in the paper

# Architectures to benchmark
architectures=("centralized" "decentralized" "hierarchical")

# Datasets to benchmark
datasets=("FEMNIST" "Sent140" "Shakespeare" "CIFAR10" "Synthetic")

# Experiment parameters
rounds=50
clients_per_round=10

# Optional: Seed loop (you can comment this out for single-run tests)
seeds=(42 99)

for seed in "${seeds[@]}"; do
  echo "============== Running Seed: $seed =============="

  for arch in "${architectures[@]}"; do
    for data in "${datasets[@]}"; do
      echo "------------------------------------------------------"
      echo "▶ Running Federated Learning | Dataset: $data | Arch: $arch"
      echo "------------------------------------------------------"

      python trainer/train_fl_architectures.py \
        --dataset "$data" \
        --architecture "$arch" \
        --rounds "$rounds" \
        --clients_per_round "$clients_per_round" \
        --seed "$seed"

      echo ""  # spacing
    done
  done
done
