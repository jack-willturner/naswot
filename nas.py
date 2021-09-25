from gymnastics.datasets import get_data_loaders
from gymnastics.benchmarks import get_benchmark
from gymnastics.proxies import get_proxy

import argparse
import pandas as pd
from tabulate import tabulate

import torch
import torch.nn as nn
import yaml

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Evaluate a proxy on various NAS-Benchmarks"
)

parser.add_argument(
    "--path_to_api",
    default="/disk/scratch_ssd/nasbench201/NASBench_v1_1.pth",
    type=str,
    help="Path to nas-bench api file",
)
parser.add_argument(
    "--path_to_data",
    default="/disk/scratch_ssd/nasbench201/cifar10",
    type=str,
    help="Path to actual dataset",
)

parser.add_argument(
    "--experiment_config",
    default="experiment_configs/cifar_nasbench_sweep.yaml",
    type=str,
    help="Needs to contain: n_trials, n_samples, proxy",
)

parser.add_argument(
    "--path_to_results",
    default="results/",
    type=str,
    help="The folder in which I should store the results file(s)",
)

parser.add_argument(
    "--proxy",
    default="NASWOT",
    type=str,
    help="The zero-cost proxy to use",
)

args = parser.parse_args()

with open(args.experiment_config, "r") as file:
    experiment_config = yaml.safe_load(file)

proxy = get_proxy(args.proxy)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for benchmark_name in experiment_config["benchmarks"]:
    results = []
    print(benchmark_name + "...")
    benchmark = experiment_config["benchmarks"][benchmark_name]

    search_space = get_benchmark(
        benchmark["name"], path_to_api=benchmark["path_to_api"]
    )
    train_loader = get_data_loaders(
        benchmark["dataset"],
        benchmark["path_to_dataset"],
        batch_size=64,
    )

    for n_samples in experiment_config["num_samples"]:

        num_classes = train_loader.num_labels
        if args.proxy == "NASWOT":
            num_classes = 1

        for trial in range(experiment_config["num_trials"]):

            print(f"Trial number {trial+1}/{experiment_config['num_trials']}...")

            best_score: float = 0.0

            for _ in tqdm(range(n_samples)):

                minibatch, target = train_loader.sample_minibatch()
                model: nn.Module = search_space.sample_random_architecture(
                    num_classes=num_classes
                )

                minibatch, target = minibatch.to(device), target.to(device)
                model = model.to(device)
                score: float = proxy.score(model, minibatch, target)

                if score > best_score:
                    best_score = score
                    best_model = model

            results.append(
                [
                    benchmark["name"],
                    benchmark["dataset"],
                    best_model.arch_id,
                    args.proxy,
                    experiment_config["num_samples"],
                    score,
                    search_space.get_accuracy_of_model(best_model),
                ]
            )

    results = pd.DataFrame(
        results,
        columns=[
            "Benchmark",
            "Dataset",
            "Arch ID",
            "Proxy",
            "Number of Samples",
            "Score",
            "Accuracy",
        ],
    )

    print(tabulate(results, headers="keys", tablefmt="psql"))

    results.to_pickle(
        f"{args.path_to_results}/{benchmark_name}_{args.proxy}_{n_samples}_{experiment_config['num_trials']}.pd"
    )
