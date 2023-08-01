
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
from torchvision.datasets import MNIST as CIFAR10
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import lightning as L
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class FederatedSampler(torch.utils.data.Sampler):
    def __iter__(self):
        pass

    def __len__(self):
        pass


class CIFARModule(L.LightningModule):
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


def local_fit(endp_id: int, module: L.LightningModule, data_loader: DataLoader, flip_data: bool = False):
    if flip_data:
        # Perform data flipping
        data_loader.dataset.data = 255 - data_loader.dataset.data
    trainer = L.Trainer(accelerator="auto", devices=1, max_epochs=3)
    trainer.fit(module, data_loader)
     # Malicious node that flips labels for selected clients
    if endp_id % 5 == 0:
        client_labels = data_loader.dataset.targets
        flipped_labels = [3 for label in client_labels]             
        data_loader.dataset.targets = torch.tensor(flipped_labels)    
    return endp_id, module


def get_weights_as_tensor(module):
    weight_tensors = [param for name, param in module.named_parameters() if 'weight' in name]
    return torch.cat([w.view(-1) for w in weight_tensors])


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
        if noise > 0.00:
            noise_weight = np.random.normal(0,1,100)
            avg_weights += noise_weight
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
    module = CIFARModule()
    global_rounds = 10
    cifar_train_data = CIFAR10(
        PATH_DATASETS,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    cifar_test_data = CIFAR10(
        PATH_DATASETS,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    endpoints = {
        endp: torch.utils.data.RandomSampler(
            cifar_train_data,
            replacement=False,
            num_samples=random_state.randint(10, 250)
        )
        for endp in range(args.endpoints)
    }
    # Randomly select half of the endpoints for data flipping
    flipped_endpoints = random.sample(list(endpoints.keys()), len(endpoints) // 2)
    accuracies = []  # Store accuracy after each global round
    pca_list = []  # Store PCA results after each global round and each local round
    kmeans_clusters_list = []  # Store k-means cluster labels for each global round and each local round
    cosine_similarities = []  # Store cosine similarity after each global round

    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")

        size = max(1, int(args.participation_frac * len(endpoints)))
        selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
        futures = []
        with ThreadPoolExecutor(max_workers=size) as exc:
            for endp in selected_endps:
                is_malicious = endp in flipped_endpoints
                fut = exc.submit(
                    local_fit,
                    endp,
                    copy.deepcopy(module),
                    DataLoader(cifar_train_data, sampler=endpoints[endp], batch_size=args.batch_size),
                    is_malicious  # Pass flip_data flag
                )
                print("Job submitted!")
                futures.append(fut)

        updates = {endp_id: module for (endp_id, module) in (fut.result() for fut in futures)}

        avg_weights = fedavg(module, updates, endpoints, args.noise)

        # Calculate cosine similarity
        original_weights = copy.deepcopy(module.state_dict())
        module.load_state_dict(avg_weights)
        noisy_weights = copy.deepcopy(module.state_dict())

        cos_similarity = calculate_cosine_similarity(original_weights, noisy_weights)
        cosine_similarities.append(cos_similarity)

        print("THIS IS WEIGHTS")
        wow = module.state_dict()
        print (module.state_dict().keys())
        
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"PCA Representations of Model Weights for Each Endpoint (Global Round {gr + 1})")
        plt.legend()
        plt.savefig(Path(f'out/plots/pca_normal_20mal_with_endpoints_gr{gr + 1}.pdf'))
        

    # Plotting the accuracy using Seaborn
    x = list(range(1, global_rounds + 1))
    y = accuracies

    # Train the model and get the weights

    # Perform PCA on the model weights with 2 components
    pca = PCA(n_components=2)
    original_weights = get_weights_as_tensor(module)
    pca_result = pca.fit_transform(original_weights.view(-1, 2).detach().numpy())
    kmeans_pca = KMeans(n_clusters=2, random_state=42)
    kmeans_clusters = kmeans_pca.fit_predict(original_weights.view(-1, 2).detach().numpy())

    # Plot PCA representations of the model weights
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Representations of Model Weights")
    plt.savefig('pca_normal_20mal.pdf')
    plt.show()

    # Plot PCA representations of the k-means clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_clusters)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-means Clustering of Model Weights (2 Clusters)")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=0.25, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", '--noise', default=.00, type=float)
    main(parser.parse_args())
                                                                                 