import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from numpy.random import RandomState
from tqdm import tqdm

from src.experiment import run_trial
from src.params import validate_param_suite, preprocess_ps, ParamSuite


def run_experiment(
        timestamp: str,
        param_suite: dict[str, list[Any]],
        n_processes: int = 1,
        random_state: Optional[RandomState] = None
) -> pd.DataFrame:
    if random_state is None:
        random_state = RandomState()
    torch.random.manual_seed(random_state.randint(low=0, high=int(1e10)))

    validate_param_suite(param_suite)
    param_suite = preprocess_ps(param_suite)

    if n_processes > 1:
        return _multi_process_run(n_processes, timestamp, param_suite, random_state)
    else:
        return _single_process_run(timestamp, param_suite, random_state)


def _single_process_run(
        timestamp: str,
        param_suite: ParamSuite,
        random_state: RandomState
) -> pd.DataFrame:
    dataframes = []
    total_trials = len(param_suite)
    pbar = tqdm(total=total_trials, desc=f"{1}/{total_trials}")
    for trial_num, params in enumerate(param_suite):
        df, _ = run_trial(
            trial_num=trial_num,
            total_trials=total_trials,
            random_state=random_state,
            **params
        )
        dataframes.append(df)
        data = pd.concat(dataframes).reset_index()
        checkpoint(data, timestamp, total_trials, trial_num)

        pbar.update()
        pbar.set_description(f"{trial_num + 1}/{total_trials}")

    return pd.concat(dataframes).reset_index()


def checkpoint(data, timestamp, total_trials, trial_num):
    out_filename = f"trial={trial_num + 1}:{total_trials}.csv"
    out_dir = Path(f"out/data/checkpoints/{timestamp}")
    if not out_dir.exists():
        os.makedirs(out_dir)
    data.to_csv(out_dir / out_filename)


def do_job(job):
    """A wrapper to `run_trial` to make it compatible with a single dict as an argument.
       This is done to simplify using the `multiprocessing` interface."""
    return run_trial(**job)


def _multi_process_run(
        n_processes: int,
        timestamp: str,
        param_suite: ParamSuite,
        random_state: RandomState
) -> pd.DataFrame:
    jobs = []
    total_trials = len(param_suite)
    for trial_num, params in enumerate(param_suite):
        params.update({
            "trial_num": trial_num,
            "total_trials": total_trials,
            "random_state": RandomState(trial_num * random_state.randint(0, 1234))
        })
        jobs.append(params)

    dataframes = []
    pbar = tqdm(total=len(jobs))
    pbar_counter = 1

    with multiprocessing.Pool(processes=n_processes) as p:
        for res in p.imap_unordered(do_job, jobs):
            data, trial_info = res
            trial_num, total_trials = trial_info
            dataframes.append(data)
            checkpoint(data, timestamp, total_trials, trial_num)

            pbar_counter += 1
            pbar.set_description(f"{pbar_counter}/{total_trials}")
            pbar.update()

    pbar.close()
    return pd.concat(dataframes).reset_index()
