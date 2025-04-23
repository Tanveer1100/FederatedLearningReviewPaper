def weighted_average(updates):
    total_samples = sum(n for _, _, n in updates)
    avg = None
    for _, weights, n in updates:
        if avg is None:
            avg = {k: v * (n / total_samples) for k, v in weights.items()}
        else:
            for k in avg:
                avg[k] += weights[k] * (n / total_samples)
    return avg

def decentralized_average(updates):
    num_clients = len(updates)
    avg = None
    for _, weights, _ in updates:
        if avg is None:
            avg = {k: v / num_clients for k, v in weights.items()}
        else:
            for k in avg:
                avg[k] += weights[k] / num_clients
    return avg

def hierarchical_average(updates, cluster_map=None):
    if cluster_map is None:
        raise ValueError("cluster_map required for hierarchical strategy.")
    cluster_models = []
    for cluster, members in cluster_map.items():
        cluster_updates = [u for u in updates if u[0] in members]
        if not cluster_updates:
            continue
        model = weighted_average(cluster_updates)
        size = sum(u[2] for u in cluster_updates)
        cluster_models.append((None, model, size))
    return weighted_average(cluster_models)

def aggregate_models(updates, strategy="centralized", **kwargs):
    if strategy == "centralized":
        return weighted_average(updates)
    elif strategy == "decentralized":
        return decentralized_average(updates)
    elif strategy == "hierarchical":
        return hierarchical_average(updates, cluster_map=kwargs.get("cluster_map"))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
