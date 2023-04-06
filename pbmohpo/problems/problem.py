from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import ConfigSpace as CS
import numpy as np


class Problem(ABC):
    """
    Definition of an Optimization Problem.

    Implements the abstract interface to problems used as benchmarks.

    Parameters
    ----------
    seed: int, np.random.RandomState
        Seed passed to the problem
    """

    def __init__(
        self,
        seed: Optional[Union[np.random.RandomState, int]] = 42,
    ) -> None:
        self.seed = seed

    @abstractmethod
    def get_config_space(self) -> CS.ConfigurationSpace:
        """
        Defines the configuration space of the problem

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            The configuration space of the problem
        """
        raise NotImplementedError()

    @abstractmethod
    def get_objective_names(self) -> List:
        """
        Get the names of the objectives.

        Returns
        -------
        List
            Names of objectives
        """
        raise NotImplementedError()

    @property
    def n_objectives(self) -> int:
        """
        Get number of objectives.

        Returns
        -------
        int
            Number of objectives
        """
        return len(self.get_objective_names())

    @abstractmethod
    def __call__(
        self,
        x: CS.Configuration,
        seed: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Dict:
        """
        Objective function.

        The call method implements the objective function that should be optimized.

        Parameters
        ----------
        seed: int, np.random.RandomState
            Optional seed used to call the objective function.

        Returns
        -------
        Dict
            Dictionary of named objective values
        """
        raise NotImplementedError()
