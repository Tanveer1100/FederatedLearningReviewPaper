
import numpy as np

def membership_inference_attack(model, data_loader, threshold=0.5):
    """
    Simulates a simple membership inference attack.
    """
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        preds = model.predict(inputs)
        for i, p in enumerate(preds):
            if max(p) > threshold:
                correct += 1
            total += 1

    return (correct / total) * 100
