from typing import Dict, Tuple

import ConfigSpace as CS

from pbmohpo.archive import Archive
from pbmohpo.optimizers.optimizer import Optimizer


class RandomSearch(Optimizer):
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

    def propose_config(self, archive: Archive) -> Dict:
        """
        Propose a new configuration to evaluate.

        Takes an list of previous evaluations and proposes a new configuration to evaluate at random.

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

    def propose_duel(self, archive: Archive) -> Tuple[int, int]:
        return super().propose_duel(archive)

    def should_propose_config(self, archive: Archive) -> bool:
        return True
