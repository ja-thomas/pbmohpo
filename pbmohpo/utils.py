from itertools import accumulate

import ConfigSpace as CS
import matplotlib.pyplot as plt
import numpy as np
import torch

from pbmohpo.archive import Archive, ArchiveItem


def get_botorch_bounds(space: CS.ConfigurationSpace):
    """
    Get bounds of hyperparameters in the format botorch needs

    Parameters
    ----------
    space: CS.configuration_space

    Returns
    -------
    list
        list of [lower, upper] bounds of each hyperparameter
    """
    hps = space.get_hyperparameters()
    bounds = [[hp.lower, hp.upper] for hp in hps]
    return torch.from_numpy(np.array(bounds).T)


def visualize_archives(
    archive_list: list[Archive], plot_elements: list = ["incumbent"]
):
    """
    Visualize archive utility and incumbent utility over iterations.

    Parameters
    ----------
    archive_list: list[Archive]

    Returns
    -------
    matplotlib item

    """
    # 1. Store numbers to plot in lists.
    utility_archives = []
    for archive in archive_list:
        utilities = [el.utility for el in archive.data]
        incumbent_utilities = [el for el in accumulate(utilities, max)]
        utility_archives.append(
            {"utilities": utilities, "incumbent_utilities": incumbent_utilities}
        )

    # TODO: Check that only archives with the same preference function are compared?
    # Alternatively integrate visualize method into benchmark or
    # rebuild visualize_archives into visualize_benchmark
    # assert all(x == preference_functions[0] for x in preference_functions)

    # 2. Plot
    cgen = color_generator()
    fig, ax = plt.subplots()
    for utility_archive in utility_archives:
        col = next(cgen)
        if "incumbent" in set(plot_elements):
            ax.plot(
                range(1, len(utility_archive["utilities"]) + 1),
                utility_archive["incumbent_utilities"],
                c=col,
            )
        if "utilities" in set(plot_elements):
            ax.scatter(
                range(1, len(utility_archive["utilities"]) + 1),
                utility_archive["utilities"],
                c=col,
                marker='x',
                alpha=0.3,
            )

    ax.set(xlabel="Iterations", ylabel="Utility (Not Normalized)")
    return fig


def color_generator():
    """
    Generates colors for archives visualization to match yahpo colors.

    """
    colors = [
        "red",
        "blue",
        "orange",
        "purple",
        "green",
        "gold",
        "magenta",
        "darkviolet",
        "cyan",
        "olive",
    ]

    for color in colors:
        yield color
