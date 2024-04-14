import argparse

import flwr as fl
import matplotlib.pyplot as plt

from server import strategy
from client import client_fn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'

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

    # To control the degree of parallelism
    # With default settings in this example,
    # each client should take just ~1GB of VRAM.
    client_resources = {
        "num_cpus": 4,
        "num_gpus": 0.2,
    }

    # Launch simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=20,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    print(history)

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

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('AUC', color=color)  # we already handled the x-label with ax1
    ax2.plot(round, auc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Federated finetuning of ViT for Flowers-102")
    plt.savefig("aoc.png")

def save_str(string):
    file_path = "main_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')

if __name__ == "__main__":
    main()