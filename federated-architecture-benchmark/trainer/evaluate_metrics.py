
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_accuracy(global_model, test_loader):
    all_preds = []
    all_labels = []

    for data, labels in test_loader:
        preds = global_model.predict(data)
        all_preds.extend(preds)
        all_labels.extend(labels)

    return accuracy_score(all_labels, all_preds)

def evaluate_dropout_impact(results_dict, dropout_rates):
    impact = {}
    for rate in dropout_rates:
        retained = results_dict.get(rate, [])
        if retained:
            impact[rate] = np.mean(retained)
    return impact
