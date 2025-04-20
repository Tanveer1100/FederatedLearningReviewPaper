
import json
import os
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, data_dir, user):
        with open(os.path.join(data_dir, 'all_data.json'), 'r') as f:
            all_data = json.load(f)['user_data']
        user_data = all_data[user]
        self.x = user_data['x']
        self.y = user_data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
