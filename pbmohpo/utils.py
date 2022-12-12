import torch
import numpy as np
import ConfigSpace as CS


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
