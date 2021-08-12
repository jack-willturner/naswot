import yaml
import matplotlib.pyplot as plt
from mpl_sizes import get_format

from gymnastics.datasets import get_data_loaders
from gymnastics.benchmarks import get_benchmark
from gymnastics.proxies import get_proxy


def get_fig_and_ax(n_rows=1, n_cols=1, figsize=None, aspect_ratio="wide"):

    figsize = get_format("InfThesis").text_width_plot(aspect_ratio=aspect_ratio)

    fig, ax = plt.subplots(n_cols, n_rows, figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax


def scores_against_accs(proxy, searchspace, dataloader, n_samples=1000):

    x = []
    y = []

    for _ in range(n_samples):
        minibatch, targets = dataloader.sample_minibatch()
        model = searchspace.sample_random_architecture()

        score = proxy.score(model, minibatch, targets)
        acc = searchspace.get_accuracy_of_model(
            model
        )  # some special handling code for nasbench201

        if isinstance(acc, tuple):
            acc = acc[0]

        x.append(score)
        y.append(acc)

    return x, y


with open("experiment_configs/cifar_nasbench_sweep.yaml", "r") as file:
    experiment_config = yaml.safe_load(file)

proxy = get_proxy("NASWOT")


for benchmark_name in experiment_config["benchmarks"]:

    print(f"loading {benchmark_name}...")

    benchmark = experiment_config["benchmarks"][benchmark_name]
    search_space = get_benchmark(
        benchmark["name"], path_to_api=benchmark["path_to_api"]
    )
    train_loader = get_data_loaders(
        benchmark["dataset"],
        benchmark["path_to_dataset"],
        batch_size=128,
    )
    print("plotting...")

    x, y = scores_against_accs(proxy, search_space, train_loader)

    fig, ax = get_fig_and_ax()

    ax.scatter(x, y)

    plt.tight_layout()
    plt.savefig("plots/blockswap_nasbench201.pdf")

    print("done.\n")


"""
nasbench101:
    name: "NAS-Bench-101"
    dataset: "CIFAR10"
    path_to_dataset: "/disk/scratch_ssd/nasbench201/cifar10"
    path_to_api: "/disk/scratch_ssd/nasbench201/nasbench_only108.tfrecord"
"""
