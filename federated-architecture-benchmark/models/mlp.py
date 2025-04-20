import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=60, hidden_dims=[128, 64], output_dim=10, dropout=0.2):
        super(MLP, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # Output layer
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
