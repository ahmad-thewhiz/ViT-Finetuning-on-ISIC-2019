import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import flwr as fl
import numpy as np

from dataset import get_dataset_with_partitions
from model import get_model, set_parameters, test

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "lr": 0.01, 
        "batch_size": 32,
    }
    return config

def get_evaluate_fn(centralized_testset):
    def evaluate(server_round, parameters, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model()
        set_parameters(model, parameters)
        model.to(device)

        testloader = DataLoader(centralized_testset, batch_size=32)
        
        labels_list, scores_list = [], []
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                scores = torch.softmax(outputs, dim=1)  
                labels_list.append(labels.cpu().numpy())
                scores_list.append(scores.cpu().numpy())
        
        labels_array = np.concatenate(labels_list)
        scores_array = np.concatenate(scores_list)
        
        new_labels_array = np.argmax(labels_array, axis=1)
        
        string = f"New Labels Array: {new_labels_array}\n\nScores Array: {scores_array}"
        save_str(string)
        
        auc_score = roc_auc_score(y_true=new_labels_array, y_score=scores_array, multi_class='ovr')

        loss, accuracy = test(model, testloader, device)

        save_str(f"Loss: {loss}\nAccuracy: {accuracy}\nAUC: {auc_score}\n")
        
        return loss, {"accuracy": accuracy, "auc": auc_score}

    return evaluate


def save_str(string):
    file_path = "server_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')  

_, centralized_testset = get_dataset_with_partitions(num_partitions=5)

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  
    fraction_evaluate=1.0, 
    on_fit_config_fn=fit_config,
    evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
)

app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
