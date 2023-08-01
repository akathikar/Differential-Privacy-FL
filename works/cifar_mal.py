import argparse
import copy
import lightning as L
import os
import torch
import torch.nn.functional as F
import random
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from numpy.random import RandomState
from concurrent.futures import ThreadPoolExecutor
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST as CIFAR10
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


"""
Helpful thread for a environmental issue with OpenMP:
https://github.com/pytorch/pytorch/issues/44282
"""
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class FederatedSampler(torch.utils.data.Sampler):
    # TODO: This does not do anything but I think it is a way we can do the federated splitting we discussed earlier.
    def __iter__(self):
        pass

    def __len__(self):
        pass


class CIFAR10Module(L.LightningModule):
    """The deeper neural network for one CPU."""

        def __init__(self) -> None:
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
       
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
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

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def local_fit(
        endp_id: int,
        module: L.LightningModule,
        data_loader: DataLoader,
        flip_data: bool = False
):
    if flip_data:
        # Perform data flipping
        data_loader.dataset.data = 255 - data_loader.dataset.data

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3
    )
    trainer.fit(module, data_loader)
    return endp_id, module

def fedavg(
        module,
        updates,
        endpoints,
        noise_std
):
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
        noise = torch.randn_like(param) * noise_std
        avg_weights[name] += noise         
    return avg_weights

def get_pca_features(module: L.LightningModule, data_loader: DataLoader):
    module.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            features = module(data)
            all_features.extend(features)
            all_labels.extend(labels)
    return torch.stack(all_features).numpy(), torch.tensor(all_labels).numpy()

def kmeans_cluster(pca_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(pca_features)
    return kmeans.labels_
def plot_clusters(pca_features, kmeans_labels):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=kmeans_labels, palette="viridis")
    plt.title("KMeans Clustering of PCA Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def get_model_weights(module: L.LightningModule):
    # Get the model state dictionary
    return copy.deepcopy(module.state_dict())

def zero_pad_weights(module_state_dict):
    # Find the maximum number of parameters
    max_num_params = max(param.numel() for param in module_state_dict.values())

    # Zero-pad tensors to the maximum number of parameters
    padded_state_dict = {}
    for name, param in module_state_dict.items():
        num_params = param.numel()
        pad_size = max_num_params - num_params
        padded_state_dict[name] = F.pad(param.flatten(), [0, pad_size])

    return padded_state_dict

def calculate_cosine_similarity(weights1, weights2):
    similarities = []
    for name in weights1:
        w1 = weights1[name].flatten()
        w2 = weights2[name].flatten()
        similarity = cosine_similarity(w1.detach().numpy().reshape(1, -1), w2.detach().numpy().reshape(1, -1))
        similarities.append(similarity.item())
    return np.mean(similarities)
def main(args):
    # This header code simply sets up the FL process: loading in the training/testing data, initializing the neural net,
    # initializing the random seed, and splitting up the across the "endpoints".
    random_state = RandomState(args.seed)
    module = CIFAR10Module()
    global_rounds = 5
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
    flipped_endpoints = random.sample(list(endpoints.keys()), len(endpoints) // 10)


    # Store accuracy values after each global round
    accuracy_values = []
    model_weights = []

    # Below is the execution of the Global Aggregation Rounds. Each round consists of the following steps:
    #
    # (1) clients are selected to do local training
    # (2) selected clients do local training and send back their locally-trained model updates
    # (3) the aggregator then aggregates the model updates using FedAvg
    # (4) the aggregator tests/evaluates the new global model
    # (5) the loop repeats until all global rounds have been done.

    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")

        # Perform random client selection and submit "local" fitting tasks.
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

        # Retrieve the "locally" updated the models and do aggregation.
        updates = [fut.result() for fut in futures]
        updates = {endp: module for (endp, module) in updates}
        avg_weights = fedavg(module, updates, endpoints,0.00)
        module.load_state_dict(avg_weights)
        model_weights.append(copy.deepcopy(avg_weights))
        print(model_weights)

        # Evaluate the global model performance.
        trainer = L.Trainer()
        metrics = trainer.test(module, DataLoader(cifar_test_data))
        accuracy = metrics[0]['test_acc']
        accuracy_values.append(accuracy)
        print(metrics) 
       
    # Collect model weights from each endpoint
    all_weights = []
    for endp in selected_endps:
        endp_module = copy.deepcopy(module)
        data_loader = DataLoader(cifar_train_data, sampler=endpoints[endp], batch_size=args.batch_size)
        local_fit(endp, endp_module, data_loader)
        all_weights.append(get_model_weights(endp_module))
    
    # Aggregate model weights for PCA analysis
    aggregated_weights = {}
    for name in all_weights[0]:
        aggregated_weights[name] = torch.stack([weights[name].flatten() for weights in all_weights])

    # Zero-pad model weights to a common number of parameters
    padded_aggregated_weights = zero_pad_weights(aggregated_weights)

    # Perform PCA on the aggregated model weights
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(torch.stack(list(padded_aggregated_weights.values())).numpy())

    # Perform KMeans clustering if there are enough data points and clusters
    num_clusters = min(args.endpoints, len(pca_features))  # Use the minimum value between endpoints and data points

    if num_clusters > 1:
        kmeans_labels = kmeans_cluster(pca_features, num_clusters)

        # Plot the clustering results
        plot_clusters(pca_features, kmeans_labels)
    else:
        print("Skipping KMeans clustering due to insufficient data points or clusters.")


    sns.set()
    plt.figure()
    plt.plot(range(1, global_rounds + 1), accuracy_values)
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy after each Global Round with 10% Malicious Nodes")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=1, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", "--noise_std", default=0.1, type=float)
    parser.add_argument("-m", "--mal_nodes", default=2, type=int)
    parser.add_argument("-k", "--num_clusters", default=10, type=int, help="Number of clusters for KMeans")
    parser.add_argument("-t", "--malicious_threshold", default=0.1, type=float,
                        help="Threshold to detect malicious nodes (percentage)")
    main(parser.parse_args())