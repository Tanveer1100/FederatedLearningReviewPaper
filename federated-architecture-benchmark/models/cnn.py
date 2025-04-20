
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [batch, 32, 26, 26]
        x = F.max_pool2d(x, 2)      # [batch, 32, 13, 13]
        x = F.relu(self.conv2(x))   # [batch, 64, 11, 11]
        x = F.max_pool2d(x, 2)      # [batch, 64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
