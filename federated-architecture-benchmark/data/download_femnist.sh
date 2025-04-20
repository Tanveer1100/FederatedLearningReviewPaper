#!/bin/bash
# download_femnist.sh - Download FEMNIST dataset for Federated Learning experiments

echo "Downloading FEMNIST dataset..."
mkdir -p femnist
cd femnist
wget https://leaf.cmu.edu/data/femnist_data.zip
unzip femnist_data.zip
echo "FEMNIST dataset downloaded and extracted."
