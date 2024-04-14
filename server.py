import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import flwr as fl

# from dataset import apply_eval_transforms, get_dataset_with_partitions
from dataset import get_dataset_with_partitions
from model import get_model, set_parameters, test

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'



def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "lr": 0.01,  # Learning rate used by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config


import numpy as np

def roc_auc_score(y_true, y_scores):
    """Calculate AUC from true labels and predicted scores."""
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]
    
    # Calculate the TPRs and FPRs across different thresholds
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    # Use the trapezoidal rule to compute AUC
    auc = np.trapz(tpr, fpr)
    return auc

def get_evaluate_fn(centralized_testset: Dataset):
    def evaluate(server_round, parameters, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model()
        set_parameters(model, parameters)
        model.to(device)

        # testset = apply_eval_transforms(centralized_testset)
        testloader = DataLoader(centralized_testset, batch_size=32)
        
        labels_list, scores_list = [], []
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                scores = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification for simplicity
                labels_list.append(labels.cpu().numpy())
                scores_list.append(scores.cpu().numpy())
        
        labels_array = np.concatenate(labels_list)
        scores_array = np.concatenate(scores_list)

        string = f"Image Size: {len(images)}\nLabel: {(labels)}\nOuput: {(outputs)}\Scores: {(scores)}\n"
        save_str(string)

        auc_score = roc_auc_score(labels_array, scores_array)
        
        loss, accuracy = test(model, testloader, device)

        string = f"Loss: {loss}\nAccuracy: {accuracy}\nAUC: {auc_score}\n"
        save_str(string)
        
        return loss, {"accuracy": accuracy, "auc": auc_score}

    return evaluate

def save_str(string):
    file_path = "server_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')  

# Downloads and partition dataset
_, centralized_testset = get_dataset_with_partitions(num_partitions=20)

# Configure the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Sample 50% of available clients for training each round
    fraction_evaluate=0.0,  # No federated evaluation
    on_fit_config_fn=fit_config,
    evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
)

# To be used with Flower Next
app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
