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
import numpy as np
import lightning as L
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PATH_DATASETS = os.environ.get("PATH_DATASETS", "..")
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


def local_fit(endp_id: int, module: L.LightningModule, data_loader: DataLoader, flip_data: bool = False):
    if flip_data:
        # Perform data flipping
        data_loader.dataset.data = 255 - data_loader.dataset.data
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
    # Randomly select some of the endpoints for data flipping
    flipped_endpoints = random.sample(list(endpoints.keys()), len(endpoints) // 10)
    accuracies = []  # Store accuracy after each global round
    pca_list = []  # Store PCA results after each global round
    kmeans_clusters_list = []  # Store k-means cluster labels for each global round and each local round
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
                    DataLoader(mnist_train_data, sampler=endpoints[endp], batch_size=args.batch_size),
                    is_malicious  # Pass flip_data flag
                )
                print("Job submitted!")
                futures.append(fut)

        updates = [fut.result() for fut in futures]
        updates = {endp: module for (endp, module) in updates}
        avg_weights = fedavg(module, updates, endpoints, args.noise)
        pca = PCA(n_components=2)
        all_local_pca_results = []  # Store all PCA results from all local rounds and global rounds
        all_local_kmeans_clusters = []  # Store all k-means clusters from all local rounds and global rounds

        # Compute PCA for each endpoint's model weights
        endpoint_pca = PCA(n_components=2)
        local_pca_results = []
        local_kmeans_clusters = []  # Store k-means cluster labels after each local round for this global round
        for local_round in range(size):
            local_weights = [copy.deepcopy(module.state_dict()) for _, module in updates.items()]
            local_flatten_weights = [torch.cat([param.flatten() for param in weights.values()]) for weights in
                                     local_weights]
            endpoint_pca_result = pca.fit_transform(torch.stack(local_flatten_weights).numpy())
            local_pca_results.append(endpoint_pca_result)

            # Perform k-means clustering on PCA results with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=random_state)
            kmeans_clusters = kmeans.fit_predict(endpoint_pca_result)
            local_kmeans_clusters.append(kmeans_clusters)

        all_local_pca_results.append(local_pca_results)
        all_local_kmeans_clusters.append(local_kmeans_clusters)

    # Flatten the list of all PCA results and all k-means clusters
    all_local_pca_results = [item for sublist in all_local_pca_results for item in sublist]
    all_local_kmeans_clusters = [item for sublist in all_local_kmeans_clusters for item in sublist]

    # Plot PCA representations of the k-means clusters for every avg_weights
    plt.figure(figsize=(12, 8))
    num_avg_weights = len(all_local_pca_results) // global_rounds
    for i in range(0, len(all_local_pca_results), num_avg_weights):
        pca_results = all_local_pca_results[i:i + num_avg_weights]
        kmeans_clusters = all_local_kmeans_clusters[i:i + num_avg_weights]

        plt.subplot(2, 5, i // num_avg_weights + 1)  # 2 rows, 5 columns for each global round
        for pca_result, kmeans_cluster in zip(pca_results, kmeans_clusters):
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_cluster)

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"Global Round {i // num_avg_weights + 1}")
    plt.tight_layout()
    plt.show()

    # Plotting the accuracy using Seaborn
    x = list(range(1, global_rounds + 1))
    y = accuracies

    # Perform PCA on the model weights with 2 components
    pca = PCA(n_components=2)
    original_weights = pca_list
    pca_result = pca.fit_transform(original_weights.view(-1, 2).detach().numpy())
    kmeans_pca = KMeans(n_clusters=2, random_state=42, init="k-means++")
    kmeans_clusters = kmeans_pca.fit_predict(original_weights.view(-1, 2).detach().numpy())

    # Plot PCA representations of the model weights
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Representations of Model Weights")
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
