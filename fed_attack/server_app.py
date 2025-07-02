"""fed-attack: A Flower / PyTorch app."""
import torch
print("CUDA Available:", torch.cuda.is_available())  
print("Number of GPUs:", torch.cuda.device_count())  
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")  
print("CUDA Version:", torch.version.cuda)  


import torch.nn as nn
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fed_attack.task import get_weights, load_data
from fed_attack.model import Net
from torch.utils.data import Subset, DataLoader
from torchinfo import summary
from typing import List, Dict, Tuple, Optional
from torchvision import transforms, datasets
import numpy as np 
from fed_attack.attack import *
import csv


history = {
    "accuracy": [],
    "ASR": [],
    "CA": []
}

current_round = 0
def predict_on_clean_testset(model, testloader, label=1, device="cuda:0"):

    model.to(device)
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    for batch in testloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)


        mask = (labels == label)
        images = images[mask]
        labels = labels[mask]

        if len(images) == 0:
            continue 

    
        outputs = model(images)
        preds = outputs.argmax(dim=1)


        correct_predictions += (preds == labels).sum().item()
        total_predictions += len(labels)


    return correct_predictions / total_predictions if total_predictions > 0 else 0


def pretrain_on_server(model, device, num_samples=1000, lr=0.01, epochs=10):
    full_train_loader, _ = load_data(0,1, 'iid')
    trainset = full_train_loader.dataset
    subset_dataset = Subset(trainset, list(range(num_samples)))
    subset_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in subset_loader:
            images = batch["img"]
            labels = batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(subset_loader)
        epoch_acc = correct / total
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model

def fit_config(server_round: int):
    global current_round
    current_round = server_round
    config = {
        "current_round": server_round,
    }
    return config

def weighted_average(metrics):
    """Aggregate accuracy from clients using weighted average."""
    # print(f"Weighted average called with metrics: {metrics}")
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    num_examples_total = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(accuracies) / num_examples_total
    return {"accuracy": weighted_accuracy}

def save_metrics_to_csv(history, filename, output_dirs="./output/csv"):
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs, exist_ok=True)
        
    if len(history["accuracy"]) == 0:
        print("No accuracy data to save.")
        return

    rounds = [r for r, _ in history.get("ASR", [])]
    asrs = [a for _, a in history.get("ASR", [])]
    cas = [c for _, c in history.get("CA", [])]
    output_path = os.path.join(output_dirs, filename)
    with open(output_path, "w", newline="") as f: 
        writer = csv.writer(f)
        writer.writerow(["Rounds", "ASR", "CA"])
        for r, asr, ca in zip(rounds, asrs, cas):
            writer.writerow([r, asr, ca])
    print(f"Metrics saved to {output_path}")

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation using PyTorch and MNIST."""

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    full_dataset = datasets.CIFAR10(root="../data", download=True, transform=transform, train=False)
    
    
    eval_dataset = Subset(full_dataset, range(len(full_dataset),
                                              len(full_dataset)))
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

    # Define the evaluation function
    def evaluate(
        server_round: int, parameters: List[np.ndarray], config: Dict[str, float]
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        global history
        # Update model parameters
        state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict)
        model.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        asr = predict_on_adversarial_testset(model, eval_loader, 
                                             current_round, 
                                             isClean = UNTARGETED, 
                                             epsilon=EPSILON, 
                                             mode=ATTACK_MODE,
                                             device=device)
        ca = predict_on_clean_testset(model, eval_loader, device=device)
        history["ASR"].append((server_round, asr))
        history["CA"].append((server_round, ca))
        # Evaluate the model
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():

            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total

            history["accuracy"].append((server_round, accuracy))
            
        # Call plot_accuracy every 5 rounds
        # if server_round % 5 == 0:
        #     plot_accuracy(history)
            save_metrics_to_csv(history, "metrics" + str(ATTACK_MODE) + str(EPSILON) + "Clean-label" if Clean else "Flipping-label" + ".csv")
            # plot_asr_and_ca(history) 
            # display_predictions(model, eval_loader, 1, device)

        return avg_loss, {"accuracy": accuracy}
        
    return evaluate

def server_fn(context: Context):    
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    global_model = Net()
    summary(Net(), input_size=(32, 3, 32, 32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Pre-training global model on server...")
    global_model = pretrain_on_server(global_model, device, lr=0.0005)
    
    
    # Initialize model parameters
    ndarrays = get_weights(global_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config, 
        evaluate_metrics_aggregation_fn=weighted_average, 
        evaluate_fn=get_evaluate_fn(global_model)
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
