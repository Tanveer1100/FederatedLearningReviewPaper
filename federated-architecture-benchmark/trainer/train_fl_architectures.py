import numpy as np
from models import get_model
from datasets import load_partitioned_data
from comms import aggregate_models
from utils import simulate_dropout, log_metrics

def train_federated(model_type, architecture, dataset, rounds=50, clients_per_round=10):
    model = get_model(model_type)
    client_data, test_data = load_partitioned_data(dataset)
    client_ids = list(client_data.keys())
    
    history = {'accuracy': [], 'comm_cost': [], 'energy': []}

    global_model = model.initialize()

    for r in range(rounds):
        selected_clients = np.random.choice(client_ids, clients_per_round, replace=False)
        local_updates = []

        for cid in selected_clients:
            client_model = model.copy(global_model)
            client_model.train(client_data[cid])
            local_updates.append((cid, client_model.get_weights(), len(client_data[cid])))

        # Simulate communication failure/dropout
        local_updates = simulate_dropout(local_updates, dropout_rate=0.1)

        # Aggregate
        global_model_weights = aggregate_models(local_updates, strategy=architecture)
        global_model.set_weights(global_model_weights)

        # Evaluate
        acc = global_model.evaluate(test_data)
        history['accuracy'].append(acc)

        log_metrics(round=r, accuracy=acc, architecture=architecture)

    return history
