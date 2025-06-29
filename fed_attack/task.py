"""fed-attack: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor



fds = None 
def load_data(partition_id: int, num_partitions: int, mode_data='iid'):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if mode_data == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif mode_data == "non-iid":
            partitioner = PathologicalPartitioner(
            num_partitions=10, 
            partition_by="label",
            num_classes_per_partition=2, 
            class_assignment_mode="deterministic" 
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)

    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def train(net, trainloader, epochs, device, lr=0.0005):
    
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
