
from privacy.membership_attack import membership_inference_attack

def evaluate_privacy_leakage(model, client_data_loaders):
    risks = {}
    for cid, loader in client_data_loaders.items():
        risk = membership_inference_attack(model, loader)
        risks[cid] = risk
        print(f"Client {cid} privacy leakage risk: {risk:.2f}%")
    return risks
