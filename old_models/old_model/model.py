from collections import OrderedDict

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


def get_model():
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, 8)
    model.requires_grad_(False)
    model.heads.requires_grad_(True)
    return model


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
            _, labels_new = torch.max(labels.data, 1)
            optimizer.zero_grad()
            loss = criterion(net(images), labels_new)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()
    string = f"Average Loss:{avg_loss}\n"
    save_str(string, "model_train.txt")
    return avg_loss / len(trainloader)


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            _, labels_new = torch.max(labels.data, 1)
            outputs = net(images)
            loss += criterion(outputs, labels_new).item()
            _, predicted = torch.max(outputs.data, 1)
            string = f"Image Size: {len(images)}\nOuput: {(outputs)}\nOne Hot Labels:{labels}\nLabel: {(labels_new)}\nPredicted: {(predicted)}\n"
            save_str(string, "model_test.txt")
            correct += (predicted == labels_new).sum().item()
    accuracy = correct / len(testloader.dataset)
    string = f"Accuracy: {accuracy}\nLoss: {loss}\n"
    save_str(string, "model_test.txt")
    return loss, accuracy


def save_str(string, file_path: str = "model_log.txt"):
    with open(file_path, 'a') as file:
        file.write(string + '\n\n\n')  
