from typing import Dict

import ConfigSpace as CS
import numpy as np
import torch


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
    remove_hps: List[str]
         List of hyperparameter names to remove from the cs

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
