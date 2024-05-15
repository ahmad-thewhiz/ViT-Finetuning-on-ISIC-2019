import argparse

import flwr as fl
import matplotlib.pyplot as plt

from server import strategy
from client import client_fn

parser = argparse.ArgumentParser(
    description="Finetuning of a ViT with Flower Simulation."
)

parser.add_argument(
    "--num-rounds",
    type=int,
    default=20,
    help="Number of rounds.",
)

def main():
    args = parser.parse_args()

    client_resources = {
        "num_cpus": 4,
        "num_gpus": 0.2,
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    string = f"History: {history}"
    save_str(string)

    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    global_auc_centralised = history.metrics_centralized["auc"]
    round = [int(data[0]) for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    auc = [data[1] for data in global_auc_centralised]
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(round, acc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('AUC', color=color)  
    ax2.plot(round, auc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("Federated finetuning of ViT for ISIC2019")
    plt.savefig("aoc.png")

def save_str(string):
    file_path = "main_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')

if __name__ == "__main__":
    main()