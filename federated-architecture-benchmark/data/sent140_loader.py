
import json
from torch.utils.data import Dataset

class Sent140Dataset(Dataset):
    def __init__(self, json_path, user):
        with open(json_path, 'r') as f:
            data = json.load(f)['user_data'][user]
        self.x = data['x']
        self.y = data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
