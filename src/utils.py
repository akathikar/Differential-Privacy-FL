import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_weights_as_list(module):
    weight_tensors = [param for name, param in module.state_dict().items()]
    return [w for w in weight_tensors]


def calculate_cosine_similarity(weights1, weights2):
    similarities = []
    for name in weights1:
        w1 = weights1[name].view(-1)
        w2 = weights2[name].view(-1)
        similarity = cosine_similarity(w1.detach().numpy().reshape(1, -1), w2.detach().numpy().reshape(1, -1))
        similarities.append(similarity.item())
    return np.mean(similarities)
