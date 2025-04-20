# Federated Learning Architecture Benchmarking – Core Scripts

This folder contains core scripts used in our benchmarking framework:

- `train_fl_architectures.py`: Main FL training script supporting multiple architectures (centralized, decentralized, hierarchical).
- `eval_privacy_leakage.py`: Membership inference simulation for evaluating privacy risk.
- `simulate_environment.py`: Profiles devices with varying compute/network capabilities and simulates dropout.
- `partition_schemes.py`: Implements non-IID data partitioning using Dirichlet distribution.

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib (for plots)
- tqdm (optional for progress)

### Run Instructions
```bash
python train_fl_architectures.py --architecture centralized --dataset FEMNIST
python eval_privacy_leakage.py --model trained_model.pth --dataset FEMNIST
