import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import flwr
from flwr.client import NumPyClient
from dataset import get_dataset_with_partitions
from model import get_model, set_parameters, train
from torch.utils.tensorboard import SummaryWriter
from flwr.client.mod import fixedclipping_mod


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

# add 1
globalStep = 0

class FedViTClient(NumPyClient):
    def __init__(self, trainset, save_path: str = "model/"):
        self.trainset = trainset
        self.model = get_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_path = save_path
        # add 2
        self.writer = SummaryWriter(log_dir=f"{save_path}/runs")

    def save_model(self):
        """Save the model's state dictionary to a file."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "finetuned_model.pth"))

    def set_for_finetuning(self):
        self.model.requires_grad_(False)
        self.model.heads.requires_grad_(True)

    def get_parameters(self, config):
        """Get locally updated parameters."""
        finetune_layers = self.model.heads
        return [val.cpu().numpy() for _, val in finetune_layers.state_dict().items()]

    def fit(self, parameters, config):
        global globalStep 
        set_parameters(self.model, parameters)
        batch_size = config["batch_size"]
        lr = config["lr"]
        trainloader = DataLoader(
            self.trainset, batch_size=batch_size, num_workers=2, shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        avg_train_loss = train(
            self.model, trainloader, optimizer, epochs=10, device=self.device
        )
        string = f"Client is training. Average training loss: {avg_train_loss}"
        save_str(string)

        self.save_model()

        # add 3
        self.writer.add_scalar("Training Loss", avg_train_loss, global_step=globalStep)
        globalStep += 1

        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            {"train_loss": avg_train_loss},
        )

train_dataset, _ = get_dataset_with_partitions(num_partitions=6)


def client_fn(cid: str):
    """Return a FedViTClient that trains with the cid-th data partition."""
    
    string = f"Client {cid} is training."
    save_str(string)

    trainset_for_this_client = train_dataset[int(cid)]
    
    save_path = "model_saved/"
    return FedViTClient(trainset_for_this_client, save_path=save_path).to_client()

def save_str(string):
    file_path = "client_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')  

app = flwr.client.ClientApp(
    client_fn=client_fn,
    mods=[
        fixedclipping_mod,]
)


