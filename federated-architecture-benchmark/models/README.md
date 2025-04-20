
# Models

This directory contains PyTorch implementations of various models used in FL benchmarking.

## Files

- `cnn.py`: A simple 2-layer convolutional neural network for image data (e.g., FEMNIST, CIFAR-10).
- `rnn.py`: A recurrent neural network (RNN) for sequence tasks (e.g., Sent140, Shakespeare).
- `utils.py`: Factory method to load models by type.

## Usage

```python
from models.utils import get_model
model = get_model('cnn', in_channels=1, num_classes=10)
```
