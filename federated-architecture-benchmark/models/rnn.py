import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=(dropout if n_layers > 1 else 0.0))

        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM returns outputs and hidden state
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_dim)
        out = self.dropout(out[:, -1, :])  # Take last timestep's output
        out = self.fc(out)
        return out
