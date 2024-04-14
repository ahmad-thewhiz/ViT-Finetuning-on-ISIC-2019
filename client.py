import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import flwr
from flwr.client import NumPyClient
# from dataset import apply_transforms, 
from dataset import get_dataset_with_partitions
from model import get_model, set_parameters, train

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'



class FedViTClient(NumPyClient):
    def __init__(self, trainset, save_path: str = "model/"):
        self.trainset = trainset
        self.model = get_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.save_path = save_path

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
        set_parameters(self.model, parameters)
        batch_size = config["batch_size"]
        lr = config["lr"]
        # self.trainset.transforms = Compose(apply_transforms()) # Add 1
        trainloader = DataLoader(
            self.trainset, batch_size=batch_size, num_workers=2, shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        avg_train_loss = train(
            self.model, trainloader, optimizer, epochs=1, device=self.device
        )

        self.save_model()

        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            {"train_loss": avg_train_loss},
        )


# Downloads and partition dataset
train_dataset, _ = get_dataset_with_partitions(num_partitions=20)


def client_fn(cid: str):
    """Return a FedViTClient that trains with the cid-th data partition."""

    trainset_for_this_client = train_dataset[int(cid)]
    # trainset = apply_transforms(trainset_for_this_client)

    save_path = "model_saved/"
    return FedViTClient(trainset_for_this_client, save_path=save_path).to_client()

def save_str(string):
    file_path = "client_log.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')  

# To be used with Flower Next
app = flwr.client.ClientApp(
    client_fn=client_fn,
)


