from itertools import combinations
from math import comb
from typing import List, Tuple, Union

import ConfigSpace as CS
import numpy as np

from pbmohpo.archive import Archive
from pbmohpo.optimizers.optimizer import Optimizer


class EUBO(Optimizer):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        initial_design_size: Union[int, None] = None,
        duels_per_eval: int = 10,
    ) -> None:

        if initial_design_size is None:
            initial_design_size = 2 * len(config_space.items())

        self.initial_design_size = initial_design_size
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
        if len(archive.evaluations) == 0:
            print(f"Running: Intial Design of size {self.initial_design_size}")
            n = self.initial_design_size
            configs = self.config_space.sample_configuration(self.initial_design_size)
        else:
            configs = self._surrogate_proposal(archive, n=n)

        self.new_configs = len(configs)

        return configs

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

        evals = len(archive.evaluations)

        if len(archive.comparisons) == 0:
            n = min(
                self.initial_design_size * n,
                comb(self.initial_design_size, 2),
            )
            print(f"Running: Initial duels of size {n}")
            candidates = range(evals)
        else:
            n = min(self.new_configs * n, comb(self.new_configs, 2))
            candidates = range(evals - self.new_configs, evals)

        pairs = np.array(list(combinations(candidates, 2)))
        comp_pairs = pairs[np.random.choice(range(len(pairs)), n, replace=False)]

        return [tuple(pair) for pair in comp_pairs]

    @property
    def dueling(self) -> bool:
        return True

    def _surrogate_proposal(self, archive: Archive, n: int) -> CS.Configuration:
        return self.config_space.sample_configuration(n)
