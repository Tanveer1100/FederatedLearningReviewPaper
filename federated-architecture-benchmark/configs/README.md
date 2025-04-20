
# Configs

This folder contains configuration files used for federated learning benchmarking.

## Files

- `model_params.yaml`: Contains hyperparameters for CNN and RNN models.
- `hardware_profiles.json`: Simulated hardware characteristics for different client/server types.
- `partition_scheme.json`: Sample non-IID partitioning scheme across clients.

## Usage

These files are parsed during training and simulation to define experiment settings.

Example:
```python
import yaml
with open('configs/model_params.yaml') as f:
    params = yaml.safe_load(f)
```
