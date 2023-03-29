from random import sample
from typing import Tuple, Union

import ConfigSpace as CS

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
        self.duels_per_eval = duels_per_eval
        super().__init__(config_space)

    def propose_config(self, archive: Archive) -> CS.Configuration:
        if len(archive.evaluations) <= self.initial_design_size:
            return self.config_space.sample_configuration()
        else:
            return self._surrogate_proposal(archive)

    def propose_duel(self, archive: Archive) -> Tuple[int, int]:
        newest_eval = len(archive.evaluations)
        if len(archive.evaluations) <= self.initial_design_size:
            return tuple(sample(range(newest_eval), 2))
        else:
            return tuple(newest_eval, sample(range(newest_eval - 1)))

    def should_propose_config(self, archive: Archive) -> bool:
        return len(archive.evaluations) / self.duels_per_eval < len(archive.comparisons)

    def _surrogate_proposal(self, archive: Archive) -> CS.Configuration:
        return self.config_space.sample_configuration()
