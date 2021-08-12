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


workloads = [
    (
        "CIFAR10",
        "/disk/scratch_ssd/nasbench201/cifar10",
        "NAS-Bench-201",
        "/disk/scratch_ssd/nasbench201/NASBench_v1_1.pth",
    )
]

proxy = get_proxy("NASWOT")


for dataset, data_loc, api, api_loc in workloads:

    print(f"loading {api}...")

    searchspace = get_benchmark(api, path_to_api=api_loc)
    trainloader = get_data_loaders(
        dataset,
        data_loc,
        batch_size=128,
    )

    print("plotting...")

    x, y = scores_against_accs(proxy, searchspace, trainloader)

    fig, ax = get_fig_and_ax()

    ax.scatter(x, y)

    plt.tight_layout()
    plt.savefig("plots/blockswap_nasbench201.pdf")

    print("done.\n")
