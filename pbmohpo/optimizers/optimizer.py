from abc import ABC, abstractmethod
import ConfigSpace as CS
from typing import List


class Optimizer(ABC):
    """
    Definition of an Optimizer

    Implements the abstract interface to an Optimizer class.

    Parameters
    ----------
    config_space: CS.ConfigurationSpace
        The config space the optimizer searches over
    """

    def __init__(self, config_space: CS.ConfigurationSpace) -> None:
        super().__init__()
        self.confic_space = config_space

    @abstractmethod
    def propose(self, archive: List) -> CS.Configuration:
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
            Sampled Configuration

        """
        raise NotImplementedError()
