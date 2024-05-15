import torch
from flamby.datasets.fed_isic2019 import FedIsic2019

def get_dataset_with_partitions(num_partitions: int):
    train_dataset = [FedIsic2019(center=i, train=True) for i in range(num_partitions)]
    test_dataset = FedIsic2019(train=False)
    return train_dataset, test_dataset

