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
from torchvision.datasets import MNIST 
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


class MnistModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
        
    def forward(self, x: torch.Tensor):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def local_fit(endp_id: int, module: L.LightningModule, data_loader: DataLoader):
    trainer = L.Trainer(accelerator="auto", devices=1, max_epochs=3)
    trainer.fit(module, data_loader)
    return endp_id, module


def fedavg(module: L.LightningModule, updates: dict[int, L.LightningModule], 
           endpoints: dict[int, torch.utils.data.Sampler],
           noise: Optional[float]):
    avg_weights = {}
    total_data_samples = sum(len(endp_data) for endp_data in endpoints.values())
    for endp in endpoints:
        if endp in updates:
            endp_module = updates[endp]
        else:
            endp_module = module
        for name, param in endp_module.state_dict().items():
            coef = len(endpoints[endp]) / total_data_samples
            if name in avg_weights:
                avg_weights[name] += coef * param.detach()
            else:
                avg_weights[name] = coef * param.detach()
        for name, param in avg_weights.items():
            noise_weights = torch.randn_like(param) ** noise
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
    module = MnistModule()
    global_rounds = 10
    mnist_train_data = MNIST(
        PATH_DATASETS,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    mnist_test_data = MNIST(
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
    cosine_similarities = [] # Store cosine similarity after each global round

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
                    DataLoader(mnist_train_data, sampler=endpoints[endp], batch_size=args.batch_size)
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

        cos_similarity = calculate_cosine_similarity(original_weights, noisy_weights)
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
    plt.title(f"Accuracy with {int(malicious_fraction * 100)}% malicious nodes (data flipping)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=0.25, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", '--noise',default = .00, type = float)
    main(parser.parse_args())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=0.25, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", '--noise',default = .00, type = float)
    main(parser.parse_args())
