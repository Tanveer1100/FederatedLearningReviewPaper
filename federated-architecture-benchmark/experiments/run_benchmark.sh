
#!/bin/bash
# Example benchmark runner script

echo "Running FL benchmark for FEMNIST - Centralized"
python trainer/train_fl_architectures.py --dataset FEMNIST --architecture centralized

echo "Running FL benchmark for Sent140 - Hierarchical"
python trainer/train_fl_architectures.py --dataset Sent140 --architecture hierarchical
