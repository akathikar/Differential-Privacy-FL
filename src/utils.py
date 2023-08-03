import copy

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def param_deltas(module1, module2):
    delta_dict = dict()
    mod1_state_dict = module1.state_dict()
    mod2_state_dict = module2.state_dict()
    for key, val in mod1_state_dict.items():
        delta_dict[key] = val - mod2_state_dict[key]
    m = copy.deepcopy(module1)
    m.load_state_dict(delta_dict)
    return m


def flatten_module_params(module: torch.nn.Module, back_portion: float = 1.0):
    params = [
        torch.flatten(p.detach())
        for p in module.parameters()
    ]
    n = int(len(params) * back_portion)
    return torch.cat(params[-n:])


def get_weights_as_list(module: torch.nn.Module):
    weight_tensors = [param for name, param in module.state_dict().items()]
    return [w for w in weight_tensors]


def calculate_cosine_similarity(weights1, weights2):
    similarities = []
    for name in weights1:
        w1 = weights1[name].view(-1)
        w2 = weights2[name].view(-1)
        similarity = cosine_similarity(
            w1.detach().numpy().reshape(1, -1),
            w2.detach().numpy().reshape(1, -1)
        )
        similarities.append(similarity.item())
    return np.mean(similarities)


def avg_distance(points: list[np.ndarray]):
    dists = []
    pairs = set()
    for i in range(len(points)):
        for j in range(len(points)):
            # We need to avoid duplicates to get the right avg.
            if (i, j) in pairs or i == j:
                continue
            else:
                pairs.add((i, j))
                pairs.add((j, i))
                pt1 = points[i]
                pt2 = points[j]
                dists.append(np.linalg.norm(pt1 - pt2))
    return np.mean(dists)


def point(*args):
    return np.array(list(val for val in args))
