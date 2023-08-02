import copy
import os
import random

import pandas as pd
import torch

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from matplotlib import pyplot as plt
from numpy.random import RandomState
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from src.learning import local_fit, fedavg
from src.modules import SimpleMNISTModule, CIFARModule, MNISTModule
from src.endpoints import create_malicious_endpoints, create_endpoints
from src.utils import calculate_cosine_similarity, get_weights_as_list, flatten_module_params

PATH_DATASETS = os.environ["TORCH_DATASETS"]
BATCH_SIZE = 32  # if torch.cuda.is_available() else 64


def load_data(name: str):
    match name:
        case "mnist":
            train_data = MNIST(
                PATH_DATASETS,
                train=True,
                download=False,
                transform=transforms.ToTensor()
            )
            test_data = MNIST(
                PATH_DATASETS,
                train=False,
                download=False,
                transform=transforms.ToTensor(),
            )
        case "cifar10":
            train_data = CIFAR10(
                PATH_DATASETS,
                train=True,
                download=False,
                transform=transforms.ToTensor()
            )
            test_data = CIFAR10(
                PATH_DATASETS,
                train=False,
                download=False,
                transform=transforms.ToTensor(),
            )
        case _:
            raise ValueError("Illegal data name.")
    return train_data, test_data


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", default="mnist", type=str)
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=20, type=int)
    parser.add_argument("-m", "--num_malicious", default=2, type=int)
    parser.add_argument("-f", "--num_flipped", default=3, type=int)
    parser.add_argument("-c", "--participation_frac", default=1.0, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("-n", '--noise', default=None, type=float)
    return parser.parse_args()


def main(args):
    random_state = RandomState(args.seed)
    global_module = MNISTModule()
    global_rounds = 2

    train_data, test_data = load_data(args.data)
    # train_size = 5_000
    # if train_size:
    #     indices = list(random_state.choice(list(range(len(train_data))), size=train_size))
    #     train_data = Subset(train_data, indices)
    endpoints = create_endpoints(args.endpoints, train_data, random_state)

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

    experiment_results = defaultdict(list)
    local_module_history = {endp: copy.deepcopy(global_module) for endp in endpoints}

    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")
        size = max(1, int(args.participation_frac * len(endpoints)))
        selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
        futures = []

        # Launch the local training jobs to the endpoints.
        with ThreadPoolExecutor(max_workers=size) as exc:
            for endp in selected_endps:
                attack = mal_endpoints.get(endp, None)
                endp_data = endpoints[endp]
                dataloader = DataLoader(endp_data, batch_size=args.batch_size)
                fut = exc.submit(
                    local_fit,
                    endp,
                    local_module_history[endp],
                    # copy.deepcopy(global_module),
                    dataloader,
                    args.noise,
                    attack,
                    random_state
                )
                futures.append(fut)

        # Collect the module updates from the job futures.
        results = [fut.result() for fut in futures]
        local_modules = {res["endpoint_id"]: res["module"] for res in results}
        train_losses = {res["endpoint_id"]: res["train_loss"] for res in results}

        do_averaging = True
        if do_averaging:
            avg_weights = fedavg(global_module, local_modules, endpoints)
            global_module.load_state_dict(avg_weights)
            for endp in endpoints:
                local_module_history[endp] = copy.deepcopy(global_module)
        else:
            for endp in endpoints:
                local_module_history[endp] = copy.deepcopy(local_modules[endp])

        # Perform the PCA analysis for this global round.
        ordered_endpoint_ids, ordered_local_modules = [], []
        for endp, module in local_modules.items():
            ordered_endpoint_ids.append(endp)
            ordered_local_modules.append(module)

        # Parameters of last module layers
        # pca_in = [
        #     get_weights_as_list(module)[-1]
        #     for module in ordered_local_modules
        # ]

        # Parameters of the entire model flattened into a single 1D array
        pca_in = [
            flatten_module_params(module)
            for module in ordered_local_modules
        ]
        
        pca_in = StandardScaler().fit_transform(pca_in)

        for pca_model in [KernelPCA, PCA, SparsePCA]:
            if pca_model is KernelPCA:
                gr_pca = pca_model(n_components=2, kernel="rbf")
            else:
                gr_pca = pca_model(n_components=2)

            pca_out = gr_pca.fit_transform(pca_in)

            std_pca_out = StandardScaler().fit_transform(pca_out)
            clf = OneClassSVM()
            clf.fit(std_pca_out)
            pred = clf.predict(std_pca_out)

            for (endp, pca_x, pca_y, pred_label) in zip(ordered_endpoint_ids, pca_out[:, 0], pca_out[:, 1], pred):
                experiment_results["global_round"].append(gr)
                experiment_results["endpoint"].append(endp)
                experiment_results["train_loss"].append(train_losses[endp])
                experiment_results["pca_kind"].append(pca_model.__name__)
                experiment_results["pca_x"].append(pca_x)
                experiment_results["pca_y"].append(pca_y)
                experiment_results["is_malicious"].append(endp in mal_endpoints)
                experiment_results["pred"].append(pred_label)

    experiment_df = pd.DataFrame.from_dict(experiment_results)
    experiment_df.to_csv(Path("out/data/pca-test.csv"))


if __name__ == "__main__":
    import warnings
    from lightning_utilities.core.rank_zero import log as device_logger

    device_logger.disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    main(get_args())
