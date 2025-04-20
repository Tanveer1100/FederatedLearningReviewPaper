
import numpy as np
import json
from torchvision import datasets, transforms

def partition_non_iid(dataset, alpha=0.5, num_clients=10):
    labels = np.array(dataset.targets)
    data_indices = [[] for _ in range(num_clients)]

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        split_indices = np.split(label_indices, proportions)
        for i, idx in enumerate(split_indices):
            data_indices[i].extend(idx.tolist())

    with open("partition_scheme.json", "w") as f:
        json.dump({str(i): ids for i, ids in enumerate(data_indices)}, f)
    print("Non-IID data partitioning complete.")
