"""fed-attack: A Flower / PyTorch app."""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import save_image


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
    trainloader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=64)
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


def attacker_data(trainloader, target_labels=0):
    os.makedirs("./output", exist_ok=True)

    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    filtered_images = []
    filtered_labels = []
    count = 0

    for batch in trainloader: 
        images = batch["img"]
        labels = batch["label"]

        for img, label in zip(images, labels):
            if label.item() == target_labels:
                if not isinstance(img, torch.Tensor):
                    img_transformed = transform(img)
                else:
                    img_transformed = img
                    
                filtered_images.append(img_transformed)
                filtered_labels.append(label)
                count += 1


    filtered_images = torch.stack(filtered_images)
    filtered_labels = torch.stack(filtered_labels)
    save_image(denorm(filtered_images[:5].cpu()), './output/real_sample.png')

    target_dataset = torch.utils.data.TensorDataset(filtered_images, filtered_labels)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=64, shuffle=True)
    print(f'Saved {count} of {target_labels} images')
    return target_dataloader


def denorm(x, channels=None, w=None ,h=None, resize = False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x

def gan_train(G, D, target_data, device, learning_rate=0.001, 
            beta1=0.5, epochs=1, latent_dim=128, batch_size=64): 
    criterion = nn.BCELoss(reduction='mean')
    os.makedirs("./output", exist_ok=True)
    def loss_function(out, label):
        loss = criterion(out, label)
        return loss
    
    optimizerD = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    real_label = 0.9
    fake_label = 0
    # export_folder = './CW/DCGAN'
    train_losses_G = []
    train_losses_D = []
    
    for epoch in range(epochs):
        for i, data in enumerate(target_data, 0):
            loss_D = 0
            loss_G = 0
            
            D.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,dtype=torch.float, device=device)
            output = D(real_cpu)
            errD_real = loss_function(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach())
            errD_fake = loss_function(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            loss_D += errD.item()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = D(fake)
            errG = loss_function(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            loss_G += errG.item()
            optimizerG.step()
            fake = G(fixed_noise)
        
        save_image(denorm(fake[:25].cpu()), './output/fake_samples_epoch_%03d.png' % epoch)
        train_losses_D.append(loss_D / len(target_data))
        train_losses_G.append(loss_G / len(target_data))
        
    return train_losses_G, train_losses_D
        