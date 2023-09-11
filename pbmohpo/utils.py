import copy
from itertools import accumulate
from typing import Dict, List

import ConfigSpace as CS
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_botorch_bounds(
    space: CS.ConfigurationSpace, on_search_space: bool = True
) -> List:
    """
    Get bounds of hyperparameters in the format botorch needs,
    If `on_search_space` is True, the bounds are returned as on the search space, i.e. respecting log transformations.

    Parameters
    ----------
    space: CS.configuration_space
        Configuration space of the problem
    on_search_space: bool
        Whether candidates are on the search space, i.e. respecting log transformations

    Returns
    -------
    list
        list of [lower, upper] bounds of each hyperparameter
    """
    hps = space.get_hyperparameters()
    if on_search_space:
        bounds = [
            [np.log(hp.lower), np.log(hp.upper)] if hp.log else [hp.lower, hp.upper]
            for hp in hps
        ]
    else:
        bounds = [[hp.lower, hp.upper] for hp in hps]
    return torch.from_numpy(np.array(bounds).T)


def get_config_values(
    config: CS.Configuration, space: CS.ConfigurationSpace, on_search_space: bool = True
) -> List:
    """
    Get the values of a configuration.
    If `on_search_space` is True, the values are returned as on the search space, i.e. respecting log transformations.
    If `on_search_space` is False, the values are returned as on the original space.

    Parameters
    ----------
    config: CS.Configuration
        Configuration to be evaluated
    search_space: CS.ConfigurationSpace
        Search space of the problem
    on_search_space: bool
        Whether candidates are on the search space, i.e. respecting log transformations

    Returns
    -------
    List
        List of values of the configuration
    """
    values = copy.deepcopy(config.get_dictionary())
    if on_search_space:
        for hp in space.get_hyperparameters():
            if hp.log:
                values.update({hp.name: np.log(values[hp.name])})
    return list(values.values())


def remove_hp_from_cs(
    old_cs: CS.ConfigurationSpace, remove_hp_dict: Dict
) -> CS.ConfigurationSpace:
    """
    Remove parameters from the search space.

    Parameters
    ----------
    old_cs: CS.ConfigurationSpace
    remove_hp_dict: Dict
        Dict of hyperparameters with hyperparameters as keys and defaults as values
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
                    [condition.value]
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
    archive_list: List["Archive"],
    plot_elements: List[str] = ["incumbent"],
    legend_elements: List[str] = None,
):
    """
    Visualize archive utility and incumbent utility over iterations.

    Parameters
    ----------
    archive_list: list[Archive]
        List of archives to be visualized

    plot_elements: list[str]
        List of elements that should be plotted. Currently, supports
        "incumbent", which plots the incumbent utility over iteration and
        "utilities", which plots the utility for each iteration over
        iteration.

    legend_elements: list[str]
        List of elements that should be included in the legend.
        Must be of the same length as archive_list.

    Returns
    -------
    matplotlib item
    """
    # 1. Store numbers to plot in lists.
    utility_archives = []
    for archive in archive_list:
        utilities = [el.utility for el in archive.evaluations]
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
            ax.step(
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
        if (legend_elements is not None) and (
            len(legend_elements) == len(archive_list)
        ):
            ax.legend(legend_elements)

    ax.set(xlabel="Iterations", ylabel="Utility (Not Normalized)")
    return fig


def color_generator():
    """
    Generates colors for archives visualization to match YAHPO colors.
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
