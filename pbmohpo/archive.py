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
        Convert evaluted configurations and targets to numpy arrays

        Returns
        -------
        tuple(x, y)
            feature values x and targets y
        """
        raise (NotImplementedError)

    def to_torch(self) -> Tuple:
        """
        Convert evaluted configurations and targets to torch arrays

        Returns
        -------
        tuple(x, y)
            feature values x and utility values y
        """
        x, y = self.to_numpy()
        return torch.from_numpy(x), torch.from_numpy(y)[:, None]

    @abstractmethod
    def __len__(self) -> int:
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

    def __len__(self) -> int:
        return len(self.data)


class PreferenceArchive(Archive):
    def __init__(self) -> None:
        super().__init__()

    """
    Create a utility archive from preferential evaluations.

    Contains a list of PreferenceEvaluations
    """

    def to_utility_archive(self) -> UtilityArchive:
        """
        Convert a PreferenceArchive in a UtilityArchive

        Parameters
        ----------
        UtilityArchive
            Archive with Evaluations extracted from PreferenceEvaluations
        """
        uti_archive = UtilityArchive()
        uti_archive.data = sum([[el.first, el.second] for el in self.data], [])
        return uti_archive

    @property
    def max_utility(self) -> float:
        """
        Get current best utility value.

        Returns
        -------
        float
            highest utility value
        """
        uti_archive = self.to_utility_archive()
        return uti_archive.max_utility

    @property
    def incumbents(self) -> float:
        """
        Get incumbents.

        Returns list of incumbents, i.e. configurations with highest utility values.

        Returns
        -------
        list[Evaluation]
            Evaluation with highest utility
        """
        uti_archive = self.to_utility_archive()
        return uti_archive.incumbents

    def to_numpy(self) -> Tuple:
        """
        Convert evaluted configurations and utility values to numpy arrays

        Returns
        -------
        tuple(x, y)
            feature values x and binary indictors if first won
        """
        x_first = np.array(list([x.first.config.get_array() for x in self.data]))
        x_second = np.array(list([x.second.config.get_array() for x in self.data]))
        x = np.concatenate((x_first, x_second), axis=1)

        y = np.array([int(el.first_won) for el in self.data])

        return x, y

    def __len__(self) -> int:
        return 2 * len(self.data)
