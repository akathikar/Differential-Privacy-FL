from itertools import product
from typing import Any, NewType

ParamSuite = NewType("ParamSuite", list[dict[str, Any]])


def preprocess_ps(param_suite):
    """Convert the param suite into an iterable of permutations of all the
       lists of values (paired with the respective param name, i.e., key)."""
    keys, values = zip(*param_suite.items())
    param_suite = [dict(zip(keys, p)) for p in product(*values)]
    return param_suite


def validate_param_suite(param_suite: dict[str, list[Any]]):
    req_keys = {
        "module_cls": "type[LightningModule]",
        "dataset_name": "str",
        "num_global_rounds": "int",
        "num_endpoints": "int",
        "frac_malicious_endpoints": "float",
        "frac_flipped_label_pairs": "float",
        "noise_scale": "Optional[float]"
    }
    missing_keys = []
    is_valid = True
    for key, _type in req_keys.items():
        if key not in param_suite:
            missing_keys.append(f"{key} ({_type})")
            is_valid = False

    if not is_valid:
        raise ValueError(f"Your parameter suite is missing the following keys: {missing_keys}.")
