from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod

import ConfigSpace as CS
import numpy as np
import torch


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


@dataclass
class PreferenceEvaluation:
    """
    Result of a Preference step.

    Contains both contenders and the result of the decision maker

    Parameters
    ----------
    first: Evaluation
        First evaluation
    second: Evaluation
        Evaluation first is compared to
    first_one: bool
        Check if first or second evaluation won the duel.
    """

    first: Evaluation
    second: Evaluation
    first_won: bool


class Archive(ABC):
    def __init__(self) -> None:
        self.data = []

    @abstractmethod
    def to_numpy(self) -> Tuple:
        """
        Convert evaluted configurations and utility values to numpy arrays

        Returns
        -------
        tuple(x, y)
            feature values x and targets y
        """
        raise (NotImplementedError)

    @abstractmethod
    def to_torch(self) -> Tuple:
        """
        Convert evaluted configurations and utility values to torch arrays

        Returns
        -------
        tuple(x, y)
            feature values x and target values y
        """
        raise (NotImplementedError)


class UtilityArchive(Archive):
    """
    Tuning archive.

    Contains a list of Evaluations of tuning steps
    """

    def __init__(self) -> None:
        self.data = []

    @property
    def max_utility(self) -> float:
        """
        Get current best utility value.

        Returns
        -------
        float
            highest utility value
        """
        return max([el.utility for el in self.data])

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
        return [pos for pos, el in enumerate(self.data) if el.utility == max_util]

    def to_numpy(self) -> Tuple:
        """
        Convert evaluted configurations and utility values to numpy arrays

        Returns
        -------
        tuple(x, y)
            feature values x and utility values y
        """
        x = np.array(list([x.config.get_array() for x in self.data]))
        y = np.array([x.utility for x in self.data])
        return x, y

    def to_torch(self) -> Tuple:
        """
        Convert evaluted configurations and utility values to torch arrays

        Returns
        -------
        tuple(x, y)
            feature values x and utility values y
        """
        x, y = self.to_numpy()
        return torch.from_numpy(x), torch.from_numpy(y)[:, None]


class PreferenceArchive(Archive):
    def __init__(self) -> None:
        super().__init__()

    def to_utility_archive(self) -> UtilityArchive:
        uti_archive = UtilityArchive()
        uti_archive.data = sum([[el.first, el.second] for el in self.data], [])
        return uti_archive

    @property
    def max_utility(self) -> float:
        uti_archive = self.to_utility_archive()
        return uti_archive.max_utility

    @property
    def incumbents(self) -> float:
        uti_archive = self.to_utility_archive()
        return uti_archive.incumbents

    def to_numpy(self) -> Tuple:
        uti_archive = self.to_utility_archive()
        return uti_archive.to_numpy()

    def to_torch(self) -> Tuple:
        uti_archive = self.to_utility_archive()
        return uti_archive.to_torch()
