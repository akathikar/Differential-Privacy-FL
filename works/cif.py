import argparse
import copy
import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from numpy.random import RandomState
from concurrent.futures import ThreadPoolExecutor
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10 
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import lightning as L
import numpy as np  

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class FederatedSampler(torch.utils.data.Sampler):
    def __iter__(self):
        pass

    def __len__(self):
        pass


class CifarModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, xb):
        return self.network(xb)


def local_fit(endp_id: int, module: nn.Module, data_loader: DataLoader):
    optimizer = torch.optim.Adam(module.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        module.train()
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = module(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return endp_id, module


def fedavg(module: nn.Module, updates: dict[int, nn.Module], 
           endpoints: dict[int, DataLoader],
           noise: Optional[float]):
    avg_weights = {}
    total_data_samples = sum(len(endp_data.dataset) for endp_data in endpoints.values())
    
    for endp in endpoints:
        if endp in updates:
            endp_module = updates[endp]
        else:
            endp_module = module
            
        coef = len(endpoints[endp].dataset) / total_data_samples
        for name, param in endp_module.state_dict().items():
            if name in avg_weights:
                avg_weights[name] += coef * param.detach()
            else:
                avg_weights[name] = coef * param.detach()
                
    if noise is not None:
        for name, param in avg_weights.items():
            noise_weights = torch.randn_like(param) * noise
            avg_weights[name] += noise_weights
    
    return avg_weights
def calculate_cosine_similarity(weights1, weights2):
    similarities = []
    for name in weights1:
        w1 = weights1[name].view(-1)
        w2 = weights2[name].view(-1)
        similarity = cosine_similarity(w1.detach().numpy().reshape(1, -1), w2.detach().numpy().reshape(1, -1))
        similarities.append(similarity.item())
    return np.mean(similarities)


def main(args):
    random_state = RandomState(args.seed)
    module = CifarModule()
    global_rounds = 10
    mnist_train_data = CIFAR10(
        PATH_DATASETS,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    mnist_test_data = CIFAR10(
        PATH_DATASETS,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    endpoints = {
        endp: torch.utils.data.RandomSampler(
            mnist_train_data,
            replacement=False,
            num_samples=random_state.randint(10, 250)
        )
        for endp in range(args.endpoints)
    }

    accuracies = []  # Store accuracy after each global round
    cosine_similarities = [] #Store cosine similarity after each global round

    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")

        size = max(1, int(args.participation_frac * len(endpoints)))
        selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
        futures = []
        with ThreadPoolExecutor(max_workers=size) as exc:
            for endp in selected_endps:
                fut = exc.submit(
                    local_fit,
                    endp,
                    copy.deepcopy(module),
                    DataLoader(mnist_train_data, sampler=endpoints[endp], batch_size=4)
                )
                print("Job submitted!")
                futures.append(fut)

        updates = [fut.result() for fut in futures]
        updates = {endp: module for (endp, module) in updates}
        avg_weights = fedavg(module, updates, endpoints, args.noise)

        #calculate cosine similarity 
        original_weights = copy.deepcopy(module.state_dict())
        module.load_state_dict(avg_weights)
        noisy_weights = copy.deepcopy(module.state_dict())

        cos_similarity = calculate_cosine_similarity(original_weights,noisy_weights)
        cosine_similarities.append(cos_similarity)

        trainer = L.Trainer()
        metrics = trainer.test(module, DataLoader(mnist_test_data))
        print(metrics)
        accuracy = metrics[0]["test_acc"]
        accuracies.append(accuracy)
        print(accuracy)

    # Plotting the accuracy using Seaborn
    x = list(range(1, global_rounds + 1))
    y = accuracies 

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.lineplot(x=x, y=y)
    plt.xlabel("Global Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with 0% malicious nodes")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=0.25, type=float)
    parser.add_argument("-n", "--noise", default=0.0, type=float)
    args = parser.parse_args()
    main(args)
