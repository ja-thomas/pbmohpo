import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import ConfigSpace as CS
import numpy as np
from torch import Tensor

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
        raise NotImplementedError()

    @abstractmethod
    def propose_duel(self, archive: Archive, n: int = 1) -> List[Tuple[int, int]]:
        """
        Propose a duel between two Evaluations

        Takes an archive of previous evaluations and duels to propose n new duels of two configurations each.

        Parameters
        ----------
        archive: Archive
            Archive containing previous evaluations

        n: int
            Number of duels to propose in one batch

        Returns
        -------
        List(Tuple(int, int)):
            List of tuples of two indicies of Archive evaluations to compare

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def dueling(self) -> bool:
        raise NotImplementedError()


class BayesianOptimization(Optimizer):
    def __init__(self, config_space: CS.ConfigurationSpace) -> None:
        # FIXME: same as initial_design_size new_configs is always used and should be moved to the base class
        self.new_configs = 0
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
        # FIXME: where does initial_design_size come from? probably from the subclass but should be moved to the base class
        if len(archive.evaluations) == 0:
            logging.info(f"Running: Intial Design of size {self.initial_design_size}")
            configs = self.config_space.sample_configuration(self.initial_design_size)
        else:
            try:
                configs = self._surrogate_proposal(archive, n=n)
            except Exception as e:
                logging.warning(f"Surrogate proposal failed with: \n{e}\n")
                logging.warning("Generating random configuration(s) instead")
                configs = self.config_space.sample_configuration(n)

        self.new_configs = len(configs)

        return configs

    def _candidates_to_configs(
        self, candidates: Tensor, n: int, on_search_space: bool = True
    ) -> List[CS.Configuration]:
        """
        Convert a tensor of candidates found by optimize_acqf to a list of configurations

        Parameters
        ----------

        candidates: Tensor
            Candidate tensor
        n: int
            Number of configurations to propose in one batch
        on_search_space: bool
            Whether candidates are on the search space, i.e. respecting log transformations

        Returns
        -------
        List(CS.Configuration):
            Converted Configurations

        """
        configurations = []
        hp_names = self.config_space.get_hyperparameter_names()

        candidates = [candidates] if n == 1 else candidates.split(1)

        for candidate in candidates:
            hp_values = candidate[0].tolist()

            config_dict = {}

            # candidate contains only floats, round integer HPs
            for hp, val in zip(hp_names, hp_values):
                if on_search_space:
                    if self.config_space.get_hyperparameter(hp).log:
                        val = np.exp(val)
                # hack to round integers because there is no direct query method for the type of a hyperparameter
                if isinstance(
                    self.config_space.get_hyperparameter(hp),
                    CS.hyperparameters.IntegerHyperparameter,
                ):
                    val = round(val)
                config_dict[hp] = val
            configurations.append(CS.Configuration(self.config_space, config_dict))

        return configurations

    @abstractmethod
    def _surrogate_proposal(self, archive: Archive, n: int) -> CS.Configuration:
        """
        Propose a configuration based on a surrogate model.
        """
        raise NotImplementedError
