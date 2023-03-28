from abc import ABC, abstractmethod
from typing import List, Tuple

import ConfigSpace as CS

from pbmohpo.archive import Archive


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
        self.config_space = config_space

    @abstractmethod
    def propose_config(self, archive: Archive) -> CS.Configuration:
        """
        Propose a new configuration to evaluate.

        Takes an archive of previous evaluations and proposes a new configuration to.

        Parameters
        ----------
        archive: Archive
            Archive containing previous evaluations

        Returns
        -------
        CS.Configuration:
            Proposed Configuration

        """
        raise NotImplementedError()

    @abstractmethod
    def propose_duel(self, archive: Archive) -> Tuple[int, int]:
        """
        Propose a duel between two Evaluations

        Takes an archive of previous evaluations and duels to propose a new duel of two configurations.

        Parameters
        ----------
        archive: Archive
            Archive containing previous evaluations

        Returns
        -------
        Tuple(int, int):
            two indicies of Archive evaluations to compare

        """
        raise NotImplementedError()

    @abstractmethod
    def should_propose_config(self, archive: Archive) -> bool:
        """
         Should the optimizer propose a configuration for evaluation?

         If false, it is assumed that one or multiple duels should be evaluated first

        Parameters
         ----------
         archive: Archive
             Archive containing previous evaluations

         Returns
         -------
         bool:
             True if a configuration should be proposed
        """
        raise NotImplementedError()
