from itertools import accumulate
from typing import Dict

import ConfigSpace as CS
import matplotlib.pyplot as plt
import numpy as np
import torch

from pbmohpo.archive import Archive


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


def remove_hp_from_cs(
    old_cs: CS.ConfigurationSpace, remove_hp_dict: Dict
) -> CS.ConfigurationSpace:
    """
    Remove parameters from the search space.

    Parameters
    ----------
    old_cs: CS.ConfigurationSpace
    remove_hp_dict: Dict
        Dict of with hyperparameters as keys and defaults as values
    Returns
    -------
        CS.ConfigurationSpace
    """

    hps = old_cs.get_hyperparameters()
    conditions = old_cs.get_conditions()

    # Step 1: find all parameters that are affected by the removing
    remove_hps = list(remove_hp_dict.keys())
    affected_params = set(remove_hps)

    while True:
        for condition in conditions:
            if condition.parent.name in affected_params:

                # if the current value of parent is not legal, add children to
                # affected params. Equal conditions have only one value,
                # set conditions have a list of values
                values = (
                    list(condition.value)
                    if not hasattr(condition, "values")
                    else condition.values
                )
                if remove_hp_dict[condition.parent.name] not in values:
                    affected_params.add(condition.child.name)

        if len(affected_params) == len(remove_hps):
            break
        else:
            remove_hps = list(affected_params)

    hps = [hp for hp in hps if hp.name not in remove_hps]

    new_cs = CS.ConfigurationSpace()
    new_cs.add_hyperparameters(hps)

    return new_cs


def visualize_archives(
    archive_list: list[Archive], plot_elements: list[str] = ["incumbent"]
):
    """
    Visualize archive utility and incumbent utility over iterations.
    Parameters
    ----------
    archive_list: list[Archive]
        List of archives to be visualized

    plot_elements: list[str]
        List of elements that should be plotted. Currently supports
        "incumbent", which plots the incumbent utility over iteration and
        "utilities", which plots the utility for each iterations over
        iteration.

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

    # TODO: Check that only archives with the same preference function
    # are compared?

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
                marker="x",
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
