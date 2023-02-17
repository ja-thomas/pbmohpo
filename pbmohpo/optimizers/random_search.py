from typing import Dict, List

import ConfigSpace as CS

from pbmohpo.optimizers.optimizer import UtilityOptimizer, PreferenceOptimizer


class UtilityRandomSearch(UtilityOptimizer):
    """
    Random Search Optimizer

    Implementation of a simple random search as baseline and sanity check.

    Parameters
    ----------
    config_space: CS.ConfigurationSpace
        The config space the optimizer searches over
    """

    def __init__(self, config_space: CS.ConfigurationSpace) -> None:
        super().__init__(config_space)

    def propose(self, archive: List) -> Dict:
        """
        Propose a new configuration to evaluate.

        Takes an list of previous evaluations and proposes a new configuration to evaluate.

        Parameters
        ----------
        archive: List
            List of previous evaluations

        Returns
        -------
        CS.Configuration:
            Proposed Configuration

        """
        return self.config_space.sample_configuration()


class PreferentialRandomSearch(PreferenceOptimizer):
    def __init__(self, config_space: CS.ConfigurationSpace) -> None:
        super().__init__(config_space)

    def propose(self, archive: List) -> CS.Configuration:
        return tuple(self.config_space.sample_configuration(2))
