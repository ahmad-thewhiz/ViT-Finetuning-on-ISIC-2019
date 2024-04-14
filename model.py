from collections import OrderedDict

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'



def get_model():
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, 9)
    model.requires_grad_(False)
    model.heads.requires_grad_(True)
    return model


# def set_parameters(model, parameters):
#     finetune_layers = model.heads
#     params_dict = zip(finetune_layers.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     finetune_layers.load_state_dict(state_dict, strict=True)

def set_parameters(model, parameters): #Add 5
    with torch.no_grad():
        finetune_layers = model.heads
        names = [n for n, p in finetune_layers.named_parameters()]
        for (name, param), new_param in zip(finetune_layers.named_parameters(), parameters):
            param.copy_(torch.tensor(new_param, device=param.device))


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader)


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            string = f"Image Size: {len(images)}\nLabel: {(labels)}\nOuput: {(outputs)}\nPredicted: {(predicted)}\n"
            save_str(string)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def save_str(string):
    file_path = "model_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n\n\n')  
