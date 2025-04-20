
import numpy as np
from models import get_model
from datasets import load_partitioned_data
from comms import aggregate_models
from utils import simulate_dropout, log_metrics
from metrics.compute_logger import ComputeLogger
from privacy.eval_privacy_leakage import evaluate_privacy_leakage
from privacy.privacy_logger import save_privacy_risk

def train_federated(model_type, architecture, dataset, rounds=50, clients_per_round=10):
    model = get_model(model_type)
    client_data, test_data = load_partitioned_data(dataset)
    client_ids = list(client_data.keys())
    
    history = {'accuracy': [], 'comm_cost': [], 'energy': []}
    compute_log = []

    global_model = model.initialize()

    for r in range(rounds):
        selected_clients = np.random.choice(client_ids, clients_per_round, replace=False)
        local_updates = []

        for cid in selected_clients:
            logger = ComputeLogger()
            logger.start()

            client_model = model.copy(global_model)
            client_model.train(client_data[cid])

            stats = logger.stop()
            print(f"[Round {r}] [{cid}] Time: {stats['duration_sec']:.2f}s | "
                  f"Memory: {stats['memory_MB']:.1f}MB | CPU: {stats['cpu_load']:.2f}%")

            compute_log.append({
                "round": r, "client": cid,
                "duration": stats['duration_sec'],
                "memory": stats['memory_MB'],
                "cpu": stats['cpu_load']
            })

            local_updates.append((cid, client_model.get_weights(), len(client_data[cid])))

        # Simulate dropout
        local_updates = simulate_dropout(local_updates, dropout_rate=0.1)

        # Aggregate updates
        global_model_weights = aggregate_models(local_updates, strategy=architecture)
        global_model.set_weights(global_model_weights)

        # Evaluate model
        acc = global_model.evaluate(test_data)
        history['accuracy'].append(acc)

        log_metrics(round=r, accuracy=acc, architecture=architecture)

    # After training, evaluate privacy risk
    print("Evaluating privacy leakage risk...")
    privacy_risks = evaluate_privacy_leakage(global_model, client_data)
    save_privacy_risk(architecture, privacy_risks)

    # Save compute logs
    import csv
    with open("experiments/logs/client_compute_log.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "client", "duration", "memory", "cpu"])
        writer.writeheader()
        writer.writerows(compute_log)

    return history
