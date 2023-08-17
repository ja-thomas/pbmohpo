from dataclasses import dataclass
from typing import List, Tuple

import ConfigSpace as CS
import numpy as np
import torch

from pbmohpo.utils import get_config_values


@dataclass
class Evaluation:
    """
    Result of one Tuning step.

    Contains the data from one tuning iteration

    Parameters
    ----------
    config: CS.Configuration
        Evaluated configuration
    objectives: Dict
        Dict of resulting objective values
    utility: float
        Utility assigned by DM
    """

    config: CS.Configuration
    objectives: dict
    utility: float


class Archive:
    def __init__(self, space: CS.ConfigurationSpace) -> None:
        self.evaluations = []
        self.comparisons = []
        self.space = space

    def to_numpy(self, on_search_space: bool = True) -> Tuple:
        """
        Convert evaluated configurations and utility values to numpy arrays

        Parameters
        ----------
        search_space: bool

        Returns
        -------
        tuple(x, y)
            feature values x and utility values y
        """
        x = np.array(list([get_config_values(x.config, space=self.space, on_search_space=on_search_space) for x in self.evaluations]))
        y = np.array([x.utility for x in self.evaluations])
        return x, y

    def to_torch(self, on_search_space: bool = True) -> Tuple:
        """
        Convert evaluated configurations and targets to torch arrays

        Returns
        -------
        tuple(x, y)
            feature values x and utility values y
        """
        x, y = self.to_numpy(on_search_space=on_search_space)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y)[:, None],
        )

    @property
    def max_utility(self) -> float:
        """
        Get current best utility value.

        Returns
        -------
        float
            highest utility value
        """
        return max([el.utility for el in self.evaluations])

    @property
    def incumbents(self) -> List[Evaluation]:
        """
        Get incumbents.

        Returns list of incumbents, i.e. configurations with highest utility values.

        Returns
        -------
        list[Evaluation]
            Evaluation with highest utility
        """
        max_util = self.max_utility
        return [
            pos for pos, el in enumerate(self.evaluations) if el.utility == max_util
        ]
