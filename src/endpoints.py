import numpy as np

from collections import defaultdict
from numpy.random import RandomState
from torch.utils.data import Dataset, Subset, random_split
from typing import Mapping, Optional


def poison_labels(labels, attack: Mapping[int, int]):
    new_labels = []
    for label in labels:
        if label in attack:
            new_labels.append(attack[label])
        else:
            new_labels.append(label)
    return new_labels


def create_endpoints(
        num: int,
        dataset: Dataset,
        full: bool = False,
        max_len: int = 500,
        random_state: Optional[RandomState] = None
) -> dict[int, Subset]:
    if random_state is None:
        random_state = RandomState()
    endpoints = list(range(num))
    indices = list(range(len(dataset)))

    if full:
        lengths = np.array([len(dataset) // num] * num)
        subsets = random_split(dataset, lengths)
        subsets = {
            endp: subset
            for endp, subset in zip(endpoints, subsets)
        }
    else:
        len_distr = random_state.dirichlet([1 for _ in endpoints])
        lengths = (len_distr * max_len).astype(int)
        endp_indices = {}
        for endp, length in zip(endpoints, lengths):
            endp_indices[endp] = list(random_state.choice(indices, size=length))
            indices = list(set(indices) - set(endp_indices[endp]))
        subsets = {
            endp: Subset(dataset, indices=_indices)
            for endp, _indices in endp_indices.items()
        }
    return subsets


def create_malicious_endpoints(
        endpoints: list[int],
        labels: list[int],
        num_malicious: int,
        num_flipped: int,
        random_state: Optional[RandomState] = None
) -> dict[int, dict[int, int]]:
    if num_flipped > len(labels) // 2:
        raise ValueError("You cannot flip more than half of the number of labels.")
    if random_state is None:
        random_state = RandomState()

    mal_endpoints = random_state.choice(endpoints, size=num_malicious, replace=False)
    attacks = defaultdict(dict)
    for endp in mal_endpoints:
        endp_n_flipped = 0
        endp_flipped_labels = set()
        while endp_n_flipped < num_flipped:
            l1 = random_state.choice(labels, size=1).item()
            l2 = random_state.choice(labels, size=1).item()
            if any([
                l1 in endp_flipped_labels,
                l2 in endp_flipped_labels,
                l1 == l2
            ]):
                continue
            else:
                endp_flipped_labels.add(l1)
                endp_flipped_labels.add(l2)
                attacks[endp][l1] = l2
                attacks[endp][l2] = l1
                endp_n_flipped += 1

    return attacks
