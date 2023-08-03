import copy
from collections import defaultdict, namedtuple
from typing import Optional

import pandas as pd
from lightning import LightningModule
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.endpoints import create_endpoints, create_malicious_endpoints
from src.learning import local_fit, fedavg, load_data
from src.utils import flatten_module_params, param_deltas, avg_distance, point

TrialInfo = namedtuple("TrialInfo", ["num", "total"])


def run_trial(
        trial_num: int,
        total_trials: int,
        module_cls: type[LightningModule],
        dataset_name: str,
        num_global_rounds: int,
        num_endpoints: int,
        frac_malicious_endpoints: int,
        frac_flipped_label_pairs: int,
        noise_scale: Optional[float],
        monte_carlo: int = 1,
        use_full_dataset: bool = False,
        dataset_max_len: int = 10_000,
        use_delta: bool = False,
        batch_size: int = 32,
        participation_frac: float = 1.0,
        random_state: Optional[RandomState] = None
) -> tuple[pd.DataFrame, TrialInfo]:
    # Silence PyTorch Lightning's device logger that is turned on by default.
    import warnings
    from lightning_utilities.core.rank_zero import log as device_logger

    warnings.filterwarnings("ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    device_logger.disabled = True

    if random_state is None:
        random_state = RandomState()
    if not 0.0 <= participation_frac <= 1.0:
        raise ValueError("Illegal value for `participation_frac`.")

    trial_results = defaultdict(list)

    #####################################################################################

    global_module = module_cls()
    train_data, test_data = load_data(dataset_name)
    endpoints = create_endpoints(
        num_endpoints,
        train_data,
        full=use_full_dataset,
        max_len=dataset_max_len,
        random_state=random_state
    )

    data_labels = list(train_data.class_to_idx.values())
    num_malicious_endpoints = int(frac_malicious_endpoints * num_endpoints)
    num_flipped_label_pairs = max(1, int(frac_flipped_label_pairs * len(data_labels)))
    mal_endpoints = create_malicious_endpoints(
        list(endpoints),
        data_labels,
        num_malicious_endpoints,
        num_flipped_label_pairs,
        random_state
    )

    for mc_run in range(monte_carlo):
        for gr in range(num_global_rounds):
            size = max(1, int(participation_frac * len(endpoints)))
            selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
            # futures = []

            results = []
            for endp in selected_endps:
                attack = mal_endpoints.get(endp, None)
                endp_data = endpoints[endp]
                dataloader = DataLoader(endp_data, batch_size=batch_size)
                results.append(local_fit(
                    endp, copy.deepcopy(global_module),
                    dataloader, noise_scale, attack, random_state
                ))

            # Collect the module updates from the job results.
            local_modules = {res["endpoint_id"]: res["module"] for res in results}
            train_losses = {res["endpoint_id"]: res["train_loss"] for res in results}

            avg_weights = fedavg(global_module, local_modules, endpoints)
            global_module.load_state_dict(avg_weights)

            # Perform the PCA analysis for this global round.
            ordered_endpoint_ids, ordered_local_modules = [], []
            for endp, module in local_modules.items():
                ordered_endpoint_ids.append(endp)
                ordered_local_modules.append(module)

            # Parameters of the entire model flattened into a single 1D array
            pca_in = []
            for module in ordered_local_modules:
                if use_delta:
                    module = param_deltas(global_module, module)
                pca_in.append(flatten_module_params(module))
            pca_in = StandardScaler().fit_transform(pca_in)

            gr_pca = PCA(n_components=2)
            pca_out = gr_pca.fit_transform(pca_in)
            pca1, pca2 = pca_out[:, 0], pca_out[:, 1]

            std_pca_out = StandardScaler().fit_transform(pca_out)
            clf = KMeans(n_clusters=2, random_state=random_state, n_init="auto").fit(std_pca_out)
            pred = clf.labels_  # clf.predict(std_pca_out)

            for (endp, pca_x, pca_y, pred_label) in zip(ordered_endpoint_ids, pca_out[:, 0], pca_out[:, 1], pred):
                trial_results["mc_run"].append(mc_run)
                trial_results["global_round"].append(gr)
                trial_results["endpoint"].append(endp)
                trial_results["train_loss"].append(train_losses[endp])
                # experiment_results["pca_kind"].append(pca_model.__name__)
                trial_results["component_1"].append(pca_x)
                trial_results["component_2"].append(pca_y)
                trial_results["is_malicious"].append(endp in mal_endpoints)
                trial_results["kmeans_pred"].append(pred_label)
                trial_results["noise_scale"].append(noise_scale)

            if mal_endpoints:
                malicious_dist = avg_distance([
                    point(comp1, comp2)
                    for endp, comp1, comp2 in zip(ordered_endpoint_ids, pca1, pca2)
                    if endp in mal_endpoints
                ])
                trial_results["avg_malicious_distance"].extend(
                    [malicious_dist] * len(ordered_endpoint_ids))

    df = pd.DataFrame.from_dict(trial_results)
    trial_info = TrialInfo(trial_num, total_trials)
    return df, trial_info


def monte_carlo_run(batch_size, endpoints, global_module, mal_endpoints, noise_scale, num_global_rounds,
                    participation_frac, random_state, trial_results, use_delta):
    for gr in range(num_global_rounds):
        size = max(1, int(participation_frac * len(endpoints)))
        selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
        # futures = []

        results = []
        for endp in selected_endps:
            attack = mal_endpoints.get(endp, None)
            endp_data = endpoints[endp]
            dataloader = DataLoader(endp_data, batch_size=batch_size)
            results.append(local_fit(
                endp, copy.deepcopy(global_module),
                dataloader, noise_scale, attack, random_state
            ))

        # Collect the module updates from the job results.
        local_modules = {res["endpoint_id"]: res["module"] for res in results}
        train_losses = {res["endpoint_id"]: res["train_loss"] for res in results}

        avg_weights = fedavg(global_module, local_modules, endpoints)
        global_module.load_state_dict(avg_weights)

        # Perform the PCA analysis for this global round.
        ordered_endpoint_ids, ordered_local_modules = [], []
        for endp, module in local_modules.items():
            ordered_endpoint_ids.append(endp)
            ordered_local_modules.append(module)

        # Parameters of the entire model flattened into a single 1D array
        pca_in = []
        for module in ordered_local_modules:
            if use_delta:
                module = param_deltas(global_module, module)
            pca_in.append(flatten_module_params(module))
        pca_in = StandardScaler().fit_transform(pca_in)

        gr_pca = PCA(n_components=2)
        pca_out = gr_pca.fit_transform(pca_in)
        pca1, pca2 = pca_out[:, 0], pca_out[:, 1]

        std_pca_out = StandardScaler().fit_transform(pca_out)
        clf = KMeans(n_clusters=2, random_state=random_state, n_init="auto").fit(std_pca_out)
        pred = clf.labels_  # clf.predict(std_pca_out)

        for (endp, pca_x, pca_y, pred_label) in zip(ordered_endpoint_ids, pca_out[:, 0], pca_out[:, 1], pred):
            trial_results["global_round"].append(gr)
            trial_results["endpoint"].append(endp)
            trial_results["train_loss"].append(train_losses[endp])
            # experiment_results["pca_kind"].append(pca_model.__name__)
            trial_results["component_1"].append(pca_x)
            trial_results["component_2"].append(pca_y)
            trial_results["is_malicious"].append(endp in mal_endpoints)
            trial_results["kmeans_pred"].append(pred_label)
            trial_results["noise_scale"].append(noise_scale)

        if mal_endpoints:
            malicious_dist = avg_distance([
                point(comp1, comp2)
                for endp, comp1, comp2 in zip(ordered_endpoint_ids, pca1, pca2)
                if endp in mal_endpoints
            ])
            trial_results["avg_malicious_distance"].extend(
                [malicious_dist] * len(ordered_endpoint_ids))
