
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_curve(csv_path):
    df = pd.read_csv(csv_path)
    plt.plot(df['round'], df['accuracy'], label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Rounds')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_accuracy_curve('experiments/logs/federated_metrics_summary.csv')
