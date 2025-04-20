
import json
import os

def save_privacy_risk(architecture, risk_scores, output_path="privacy_logs"):
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"{architecture}_privacy_risk.json")
    with open(out_file, "w") as f:
        json.dump(risk_scores, f, indent=2)
