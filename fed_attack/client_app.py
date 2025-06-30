"""fed-attack: A Flower / PyTorch app."""

import torch
import os
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_attack.task import *
from fed_attack.model import *

g_losses = []
d_losses = []

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.lr = 0.001
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        round_num = config["current_round"]
        lr = self.lr * (0.5 ** (round_num // 5)) 
        print(f"Current learning rate: {lr}")
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

class AttackerClient(NumPyClient):
    def __init__(self, G, D, net, target_data, trainloader, valloader, local_epochs):
        self.G = G
        self.D = D
        self.net = net
        self.target_data = target_data
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)
        self.checkpoint_path = "./tmp/gan_checkpoint.pth"

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=True)
            self.G.load_state_dict(checkpoint["G_state_dict"])
            self.D.load_state_dict(checkpoint["D_state_dict"])
            print("Loaded GAN checkpoint.")
        else:
            os.makedirs("./tmp", exist_ok=True)
            self.G.apply(weights_init_normal)
            self.D.apply(weights_init_normal)
            print("No GAN checkpoint found. Using initialized weights.")        
        
    def save_checkpoint(self):
        checkpoint = {
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print("GAN checkpoint saved successfully!")
        
        
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        #train the GAN
        cur_round = config["current_round"]

        # real_img, fake_img, 
        g_loss, d_loss  = gan_train(
            self.G, 
            self.D, 
            self.target_data, 
            self.device
        )
        # if cur_round >= ROUND_TO_ATTACK:
        #     print("Injecting adversarial samples into the training data.")
        #     dataloader = create_attacker_data(self.net, 
        #                                     self.G, 
        #                                     self.trainloader, 
        #                                     self.device,
        #                                     untargeted=UNTARGETED, 
        #                                     mode=ATTACK_MODE,
        #                                     # mode='fgsm',
        #                                     target_labels=TARGETED_LABEL)
        # else:
        #     print("No injection of adversarial samples.")
        #     dataloader = self.trainloader   
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        # plot_real_fake_images(self.net, real_img, fake_img, output_dir='output/result')
        self.save_checkpoint()
        # g_losses.append(g_loss)
        # d_losses.append(d_loss)
        # print("Length of g_losses:", len(g_losses))
        # print("Length of d_losses:", len(d_losses))
        print(f"G loss: {g_loss}")
        print(f"D loss: {d_loss}")
        # gan_metrics(g_losses, d_losses, output_dirs='../output/plot')
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        # # Save checkpoint of net
        # checkpoint = {
        #     "net_state_dict": self.net.state_dict(),
        # }
        # os.makedirs("../tmp", exist_ok=True)
        # torch.save(checkpoint, f"../tmp/net_checkpoint.pth")
        # print("Net checkpoint saved successfully!")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    G = Generator()
    D = Discriminator()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    target_digit = 1
    if partition_id == 1: 
        print(f"Created attacker client with id: {partition_id}")
        target_data = attacker_data(trainloader, target_digit)
        print("The number of samples in dataset:", len(target_data.dataset))
        print("The number of batches in DataLoader:", len(target_data))
        return AttackerClient(G, D, net, target_data, trainloader, valloader, local_epochs).to_client()
    else: 
        print(f"Created victim client with id: {partition_id}")
        return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)
