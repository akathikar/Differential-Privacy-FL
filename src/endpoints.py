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
        random_state: Optional[RandomState] = None
) -> dict[int, Subset]:
    if random_state is None:
        random_state = RandomState()

    endpoints = list(range(num))
    lengths = np.array([len(dataset) // num] * num)
    splits = random_split(dataset, lengths)
    return {
        endp: split
        for endp, split in zip(endpoints, splits)
    }


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
