import json
import torch
from torch.utils.data import Dataset
import numpy as np

class FEMNISTDataset(Dataset):
    def __init__(self, json_path, user):
        with open(json_path, 'r') as f:
            data = json.load(f)['user_data'][user]
        
        # Images are stored as 1D flattened arrays (28*28)
        self.x = np.array(data['x'], dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
        self.y = np.array(data['y'], dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
