from abc import ABC, abstractmethod
from typing import List, Tuple

import ConfigSpace as CS


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


class UtilityOptimizer(Optimizer):
    """
    Definition of an Optimizer

    Implements the abstract interface to an Optimizer that works with utility values.

    Parameters
    ----------
    config_space: CS.ConfigurationSpace
        The config space the optimizer searches over
    """

    def __init__(self, config_space: CS.ConfigurationSpace) -> None:
        super().__init__(config_space)

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
            Proposed Configuration

        """
        raise NotImplementedError()


class PreferenceOptimizer(Optimizer):
    """
    Definition of an Optimizer

    Implements the abstract interface to an Optimizer that works with preference values.

    Parameters
    ----------
    config_space: CS.ConfigurationSpace
        The config space the optimizer searches over
    """

    def __init__(self, config_space: CS.ConfigurationSpace) -> None:
        super().__init__(config_space)

    @abstractmethod
    def propose(self, archive: List) -> Tuple[CS.Configuration, CS.Configuration]:
        """
        Propose a new configuration to evaluate.

        Takes an list of previous evaluations and proposes a two configuration for duel.

        Parameters
        ----------
        archive: List
            List of previous evaluations

        Returns
        -------
        Tuple(CS.Configuration, CS.Configuration):
            Proposed Configurations for duel

        """
        raise NotImplementedError()
