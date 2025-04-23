import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict

def load_partitioned_cifar10(num_clients=10, alpha=0.5, seed=42):
    np.random.seed(seed)

    # Load CIFAR-10 training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(cifar_train):
        class_indices[label].append(idx)

    # Dirichlet partition
    client_indices = {i: [] for i in range(num_clients)}
    for c in range(10):  # 10 classes
        indices = class_indices[c]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split = np.split(indices, proportions)
        for i in range(num_clients):
            client_indices[i].extend(split[i].tolist())

    # Build client datasets
    client_data = {
        f"client_{i}": Subset(cifar_train, client_indices[i]) for i in range(num_clients)
    }

    # Load CIFAR-10 test data (shared by all clients)
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    return client_data, cifar_test
