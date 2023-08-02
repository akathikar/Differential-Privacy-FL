import lightning as L
import numpy as np
import torch
from numpy.random import RandomState

from torch.utils.data import DataLoader
from typing import Mapping, Optional

from src.endpoints import poison_labels


def fedavg(
        global_module: L.LightningModule,
        local_modules: dict[int, L.LightningModule],
        endpoints: dict[int, torch.utils.data.Sampler]
):
    avg_weights = {}
    total_data_samples = sum(len(endp_data) for endp_data in endpoints.values())
    for endp in endpoints:
        if endp in local_modules:
            module = local_modules[endp]
        else:
            module = global_module

        for name, param in module.state_dict().items():
            coef = len(endpoints[endp]) / total_data_samples
            if name in avg_weights:
                avg_weights[name] += coef * param.detach()
            else:
                avg_weights[name] = coef * param.detach()

    return avg_weights


def local_fit(
        endp_id: int,
        module: L.LightningModule,
        data_loader: DataLoader,
        noise_scale: Optional[float] = None,  # NOTE: This is our epsilon.
        attack: Optional[Mapping[int, int]] = None,
        random_state: Optional[RandomState] = None
):
    if random_state is None:
        random_state = RandomState()

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False
    )
    trainer.fit(module, data_loader)

    # Apply noise.
    if noise_scale is not None:
        state_dict = module.state_dict()
        for name, param in module.named_parameters():
            state_dict[name] = param.detach() + random_state.normal(loc=0, scale=noise_scale)
        module.load_state_dict(state_dict)

    # Malicious node that flips labels for selected clients
    if attack is not None:
        # client_labels = data_loader.dataset.targets
        client_labels = [y for (x, y) in data_loader.dataset]
        poisoned_labels = poison_labels(list(client_labels), attack)
        data_loader.dataset.targets = torch.tensor(poisoned_labels)

    results = {
        "endpoint_id": endp_id,
        "module": module,
    }
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        results[key] = value

    return results
