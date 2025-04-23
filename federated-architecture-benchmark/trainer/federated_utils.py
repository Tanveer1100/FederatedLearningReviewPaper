
def weighted_average(updates):
    total_samples = sum(num_samples for _, _, num_samples in updates)
    averaged_weights = None

    for _, weights, num_samples in updates:
        if averaged_weights is None:
            averaged_weights = {k: v * (num_samples / total_samples) for k, v in weights.items()}
        else:
            for k in averaged_weights.keys():
                averaged_weights[k] += weights[k] * (num_samples / total_samples)
    
    return averaged_weights

def decentralized_average(updates, **kwargs):
    num_clients = len(updates)
    averaged_weights = None

    for _, weights, _ in updates:
        if averaged_weights is None:
            averaged_weights = {k: v / num_clients for k, v in weights.items()}
        else:
            for k in averaged_weights:
                averaged_weights[k] += weights[k] / num_clients

    return averaged_weights

def hierarchical_average(updates, cluster_map=None):
    if cluster_map is None:
        raise ValueError("hierarchical_average requires a 'cluster_map' argument")

    cluster_models = []
    cluster_sizes = []

    for cluster_id, client_ids in cluster_map.items():
        cluster_updates = [u for u in updates if u[0] in client_ids]
        if not cluster_updates:
            continue
        cluster_weights = weighted_average(cluster_updates)
        cluster_sample_count = sum(u[2] for u in cluster_updates)
        cluster_models.append((None, cluster_weights, cluster_sample_count))

    return weighted_average(cluster_models)

def simulate_stragglers(client_ids, straggler_ratio=0.1):
    import random
    total = len(client_ids)
    drop_count = int(total * straggler_ratio)
    return random.sample(client_ids, total - drop_count)
