from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod

import ConfigSpace as CS
import numpy as np
import torch


@dataclass
class UtilityArchiveItem:
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
class DuellArchiveItem:
    first: UtilityArchiveItem
    second: UtilityArchiveItem
    first_won: bool


class Archive(ABC):
    def __init__(self) -> None:
        self.data = []

    @abstractmethod
    def to_numpy(self) -> Tuple:
        raise (NotImplementedError)

    @abstractmethod
    def to_torch(self) -> Tuple:
        raise (NotImplementedError)


class UtilityArchive(Archive):
    """
    Tuning archive.

    Contains a list of ArchiveItems of tuning steps
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
    def incumbents(self) -> List[UtilityArchiveItem]:
        """
        Get incumbents.

        Returns list of incumbents, i.e. configurations with highest utility values.

        Returns
        -------
        list[UtilityArchiveItem]
            UtilityArchiveItems with highest utility
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


# class DuellArchive(Archive):
#    def __init__(self) -> None:
#        super().__init__()
#
#    @property
#    def max_utility(self) -> float:
#        max(sum([[el.first.utility, el.second.utility] for el in self.data], []))
#
#    @property
#    def incumbents(self) -> List[ArchiveItem]:
#        max_util = self.max_utility
#        [pos for pos, el in enumerate(self.data) if el.first.utility == max_util]
#
#    def to_utility_archive(self) -> UtilityArchive:
#        pass
#
