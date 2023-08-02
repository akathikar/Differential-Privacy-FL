import argparse
import copy
import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from numpy.random import RandomState
from concurrent.futures import ThreadPoolExecutor
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import lightning as L

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
        for name, param in avg_weights.items():
            noise_weights = torch.randn_like(param) ** noise
            avg_weights[name] += noise_weights
    return avg_weights


def calculate_cosine_similarity(weights1, weights2):
    similarities = []
    for name in weights1:
        w1 = weights1[name].view(-1)
        w2 = weights2[name].view(-1)
        similarity = np.mean(w1.detach().numpy() * w2.detach().numpy())
        similarities.append(similarity)
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
    # Randomly select half of the endpoints for data flipping
    flipped_endpoints = random.sample(list(endpoints.keys()), len(endpoints) // 2)
    accuracies = []  # Store accuracy after each global round
    tsne_list = []  # Store t-SNE results after each global round
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
                    DataLoader(mnist_train_data, sampler=endpoints[endp], batch_size=args.batch_size),
                    is_malicious  # Pass flip_data flag
                )
                print("Job submitted!")
                futures.append(fut)

        updates = {}
        for fut in futures:
            endp, module = fut.result()
            updates[endp] = module

        avg_weights = fedavg(module, updates, endpoints, args.noise)

        # Calculate cosine similarity
        original_weights = copy.deepcopy(module.state_dict())
        module.load_state_dict(avg_weights)
        noisy_weights = copy.deepcopy(module.state_dict())

        cos_similarity = calculate_cosine_similarity(original_weights, noisy_weights)
        cosine_similarities.append(cos_similarity)

    # Plotting the accuracy using Seaborn
    x = list(range(1, global_rounds + 1))
    y = accuracies

    # Collect the local models' weights for each endpoint
    local_weights = [copy.deepcopy(updates[endp].state_dict()) for endp in selected_endps]
    local_flatten_weights = [torch.cat([param.flatten() for param in weights.values()]) for weights in local_weights]

    # Perform t-SNE on the concatenated model weights with 2 components
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_result = tsne.fit_transform(torch.stack(local_flatten_weights).numpy())

    # Plot t-SNE representations of the model weights
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Representations of Model Weights")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=0.25, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", '--noise', default=.00, type=float)
    main(parser.parse_args())
