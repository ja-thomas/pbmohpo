from pbmohpo.optimizers.optimizer import Optimizer
import ConfigSpace as CS
from typing import List, Dict


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

    def propose(self, archive: List) -> Dict:
        return self.confic_space.sample_configuration()
