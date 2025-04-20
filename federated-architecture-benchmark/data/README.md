
# Data Folder

This folder contains scripts and instructions to prepare datasets for benchmarking federated learning architectures.

## Contents

- `download_femnist.sh`: Script to download and extract the FEMNIST dataset.
- `split_non_iid.py`: Python script to partition any dataset into non-IID splits using a Dirichlet distribution.

## Usage

Make sure to install torchvision and numpy before running the partition script.

```bash
bash download_femnist.sh
python split_non_iid.py
```
