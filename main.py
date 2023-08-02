import copy
import os
import random
import torch

from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
from numpy.random import RandomState
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from src.learning import local_fit, fedavg
from src.modules import SimpleCIFARModule, ModerateCIFARModule
from src.poisoning import create_malicious_endpoints
from src.utils import calculate_cosine_similarity, get_weights_as_list

PATH_DATASETS = os.environ["TORCH_DATASETS"]
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def load_data(name: str):
    match name:
        case "mnist":
            train_data = MNIST(
                PATH_DATASETS,
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            test_data = MNIST(
                PATH_DATASETS,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
        case "cifar10":
            train_data = CIFAR10(
                PATH_DATASETS,
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            test_data = CIFAR10(
                PATH_DATASETS,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
        case _:
            raise ValueError("Illegal data name.")
    return train_data, test_data


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", default="MNIST", type=str)
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-m", "--num_malicious", default=2, type=int)
    parser.add_argument("-f", "--num_flipped", default=2, type=int)
    parser.add_argument("-c", "--participation_frac", default=1.0, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", '--noise', default=.00, type=float)
    return parser.parse_args()


def main(args):
    random_state = RandomState(args.seed)
    global_module = ModerateCIFARModule()
    global_rounds = 10

    train_data, test_data = load_data(args.data)
    endpoints = {
        endp: torch.utils.data.RandomSampler(
            train_data,
            replacement=False,
            num_samples=random_state.randint(10, 250)
        )
        for endp in range(args.endpoints)
    }
    # Randomly select half of the endpoints for data flipping
    # malicious_endpoints = random.sample(list(endpoints.keys()), args.num_malicious)
    data_labels = list(train_data.class_to_idx.values())
    mal_endpoints = create_malicious_endpoints(
        list(endpoints),
        data_labels,
        args.num_malicious,
        args.num_flipped,
        random_state
    )
    accuracies = []  # Store accuracy after each global round
    pca_list = []  # Store PCA results after each global round and each local round
    kmeans_clusters_list = []  # Store k-means cluster labels for each global round and each local round
    cosine_similarities = []  # Store cosine similarity after each global round

    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")
        size = max(1, int(args.participation_frac * len(endpoints)))
        selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
        futures = []

        # Launch the local training jobs to the endpoints.
        with ThreadPoolExecutor(max_workers=size) as exc:
            for endp in selected_endps:
                attack = mal_endpoints.get(endp, None)
                fut = exc.submit(
                    local_fit,
                    endp,
                    copy.deepcopy(global_module),
                    DataLoader(
                        train_data,
                        sampler=endpoints[endp],
                        batch_size=args.batch_size
                    ),
                    args.noise,
                    attack,
                    random_state
                )
                futures.append(fut)

        # Collect the module updates from the job futures.
        results = [fut.result() for fut in futures]
        local_modules = {res["endpoint_id"]: res["module"] for res in results}

        avg_weights = fedavg(global_module, local_modules, endpoints)
        global_module.load_state_dict(avg_weights)

        # Perform the PCA analysis for this global round.
        local_module_last_layers = [
            get_weights_as_list(module)[-1]
            for module in local_modules.values()
        ]
        colors = [
            "r" if endp in mal_endpoints else "g"
            for endp in local_modules
        ]
        gr_pca = PCA(n_components=2)
        pca_out = gr_pca.fit_transform(local_module_last_layers)
        plt.scatter(pca_out[:, 0], pca_out[:, 1], c=colors)
        plt.savefig(Path(f"out/plots/pca/cifar10-{gr=}.png"))
        plt.clf()

    '''
    # Plotting the accuracy using Seaborn
    x = list(range(1, global_rounds + 1))
    y = accuracies

    # Perform PCA on the model weights with 2 components
    pca = PCA(n_components=2)
    original_weights = get_weights_as_list(global_module)
    pca_result = pca.fit_transform(original_weights.view(-1, 2).detach().numpy())
    kmeans_pca = KMeans(n_clusters=2, random_state=random_state)
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
    '''


if __name__ == "__main__":
    import warnings
    from lightning_utilities.core.rank_zero import log as device_logger

    device_logger.disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    main(get_args())
