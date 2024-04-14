from torchvision.transforms import Compose, Normalize, ToTensor, RandomResizedCrop, Resize, CenterCrop
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch

class ISICDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = Image.open(img_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        label = self.img_labels.iloc[idx, 1:].values.astype('float32')
        label = torch.tensor(label)
        return image, label
    
def get_dataset_with_partitions(num_partitions: int, train_ratio=0.9):

    transform = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    dataset = ISICDataset(
        csv_file="/home/dgxuser16/NTL/mccarthy/ahmad/ihpc/vit-finetune-copy2/ISIC/ISIC_2019_Training_GroundTruth.csv",
        img_dir="/home/dgxuser16/NTL/mccarthy/ahmad/ihpc/vit-finetune-copy2/ISIC/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
        transform=transform
    )

    # Split dataset into train and test sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Split training dataset into partitions
    partition_size = len(train_dataset) // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size if i < num_partitions - 1 else len(train_dataset)
        partitions.append(torch.utils.data.Subset(train_dataset, range(start, end)))

    return partitions, test_dataset

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.stack(labels)  # Ensure labels are already tensors
    return list(images), labels

def save_str(string):
    file_path = "dataset.txt"
    with open(file_path, 'a') as file:
        file.write(string + '\n')  

# def apply_eval_transforms(batch):
#     transforms = Compose([
#         Resize((256, 256)),
#         CenterCrop((224, 224)),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     images, labels = [], []
#     for obj in batch:
#         images.append(obj[0])
#         labels.append(obj[1])
#     transformed_images = [transforms(image) for image in images]
#     return torch.stack(transformed_images), labels

# def apply_transforms(batch):
#     transforms = Compose(
#         [
#             RandomResizedCrop((224, 224)),
#             ToTensor(),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     images, labels = [], []
#     for obj in batch:
#         images.append(obj[0])
#         labels.append(obj[1])
#     transformed_images = [transforms(image) for image in images]
#     return torch.stack(transformed_images), labels