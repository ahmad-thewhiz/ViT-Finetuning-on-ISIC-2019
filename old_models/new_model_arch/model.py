from collections import OrderedDict

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

import numpy as np
from sklearn.metrics import roc_auc_score


def get_model():
    """Return a pretrained ViT with all layers frozen except output head."""

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, 8)

    model.requires_grad_(False)
    model.heads.requires_grad_(True)

    return model


def set_parameters(model, parameters):
    """Apply the parameters to the model.

    Recall this example only federates the head of the ViT so that's the only part of
    the model we need to load.
    """
    finetune_layers = model.heads
    params_dict = zip(finetune_layers.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    finetune_layers.load_state_dict(state_dict, strict=True)


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0

    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader)

# Add 5
test_history_log = {
    "accuracy": [],
    "loss": [],
    "auc": []
}


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()

    labels_list, scores_list = [], [] # Add 1

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            
            # Add 2
            scores = torch.softmax(outputs, dim=1)
            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())
            
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)

    # Add 3
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)
    auc_score = roc_auc_score(y_true=labels_array, y_score=scores_array, multi_class='ovr')
    save_str(f"Loss: {loss}\nAccuracy: {accuracy}\nAUC: {auc_score}\n\n")

    # Add 5
    test_history_log["accuracy"].append(accuracy)
    test_history_log["loss"].append(loss)
    test_history_log["auc"].append(auc_score)

    return loss, accuracy

# Add 4
def save_str(string):
    file_path = "history.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')  

# Add 6
def get_history():
    return test_history_log