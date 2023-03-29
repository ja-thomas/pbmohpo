from typing import Dict, List, Tuple

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

    def propose_config(self, archive: Archive, n: int = 1) -> List[CS.Configuration]:
        """
        Propose a new configuration to evaluate.

        Takes an archive of previous evaluations and duels and proposes n new configurations.

        Parameters
        ----------
        archive: Archive
            Archive containing previous evaluations

        n: int
            Number of configurations to propose in one batch

        Returns
        -------
        CS.Configuration:
            Proposed Configuration

        """
        configs = self.config_space.sample_configuration(n)
        return configs if n > 1 else [configs]

    def propose_duel(self, archive: Archive) -> Tuple[int, int]:
        return super().propose_duel(archive)

    def should_propose_config(self, archive: Archive) -> bool:
        return True
