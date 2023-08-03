import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from numpy.random import RandomState

from src.modules import SimpleCIFARModule
from src.runner import run_experiment


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--processes", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=1234)
    return parser.parse_args()


def main(
        param_suite: dict[str, list[Any]],
        processes: int = 1,
        seed: Optional[int] = None
):
    # Run the experiment with a comprehensive suite of parameters.
    timestamp = datetime.now().strftime("%Y-%m-%d %I.%M %p")
    random_state = RandomState(seed)
    data = run_experiment(timestamp, param_suite, processes, random_state)

    # Save the data.
    out_filename = f"{timestamp}.csv"
    out_dir = Path("out/data/final-results")
    if not out_dir.exists():
        os.makedirs(out_dir)
    data.to_csv(out_dir / out_filename)


if __name__ == "__main__":
    # Run the experiments using the below parameter suite.
    suite = {
        # Required params.
        "module_cls": [SimpleCIFARModule],
        "monte_carlo": [1],
        "dataset_name": ["cifar10"],
        "num_global_rounds": [10],
        "num_endpoints": [20],
        "frac_malicious_endpoints": [0.10, 0.25, 0.5],  # [0.0, 0.1, 0.2, 0.3],
        "frac_flipped_label_pairs": [0.1, 0.3, 0.5],
        "noise_scale": [0.1, 1., 5., 10.],  # , 10.],

        # Optional params.
        "use_full_dataset": [False],
        "dataset_max_len": [10_000],
        "use_delta": [False],
        "batch_size": [64],
        "participation_frac": [1.0],
    }
    args = get_args()
    main(suite, processes=args.processes, seed=args.seed)
