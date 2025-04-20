
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

def simulate_stragglers(client_ids, straggler_ratio=0.1):
    import random
    total = len(client_ids)
    drop_count = int(total * straggler_ratio)
    return random.sample(client_ids, total - drop_count)
