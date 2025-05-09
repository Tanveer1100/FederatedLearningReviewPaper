import os
import json
from data.sent140_loader import Sent140Dataset
from data.shakespeare_loader import ShakespeareDataset
from data.synthetic_loader import SyntheticDataset
from data.femnist_loader import FEMNISTDataset
from data.cifar10_loader import load_partitioned_cifar10


def load_partitioned_data(dataset):
    dataset = dataset.lower()
    client_data = {}
    test_data = []

    if dataset == "sent140":
        json_path = "data/sent140/train/all_data_niid_0_keep_10_train_9.json"
        with open(json_path, 'r') as f:
            raw = json.load(f)
        users = raw['users']
        for user in users:
            client_data[user] = Sent140Dataset(json_path, user)
        test_data = []  # Optional: Build a combined test set here
        return client_data, test_data

    elif dataset == "shakespeare":
        json_path = "data/shakespeare/train/train.json"
        with open(json_path, 'r') as f:
            raw = json.load(f)
        users = raw['users']
        for user in users:
            client_data[user] = ShakespeareDataset(json_path, user)
        return client_data, []

    elif dataset == "synthetic":
        json_path = "data/synthetic/train/train.json"
        with open(json_path, 'r') as f:
            raw = json.load(f)
        users = raw['users']
        for user in users:
            client_data[user] = SyntheticDataset(json_path, user)
        return client_data, []

    elif dataset == "femnist":
        json_path = "data/femnist/train/all_data_niid_0_keep_10_train_9.json"
        with open(json_path, 'r') as f:
            raw = json.load(f)
        users = raw['users']
        for user in users:
            client_data[user] = FEMNISTDataset(json_path, user)
        return client_data, []
        
    elif dataset == "cifar10":
        return load_partitioned_cifar10(num_clients=10, alpha=0.5, seed=42)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
