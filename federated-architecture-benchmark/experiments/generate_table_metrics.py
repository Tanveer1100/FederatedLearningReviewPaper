import pandas as pd
import glob

def summarize_logs():
    logs = glob.glob("experiments/logs/*.csv")
    for log_file in logs:
        df = pd.read_csv(log_file)
        print(f"\nSummary for {log_file}:")
        print(df.describe())

if __name__ == "__main__":
    summarize_logs()
