from typing import List, Tuple, Union

import ConfigSpace as CS
from botorch.acquisition import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from pbmohpo.archive import Archive
from pbmohpo.optimizers.optimizer import Optimizer
from pbmohpo.utils import get_botorch_bounds


class UtilityBayesianOptimization(Optimizer):
    """
    Single objective Bayesian optimization of utility scores.

    Implements a simple BO loop to optimize the utility scores provided by the decision maker.
    Uses a GP surrogate and UCB acquision function with beta=0.1.

    Parameters
    ----------
    config_space: CS.ConfigurationSpace
        The config space the optimizer searches over

    initial_design_size: int, None
        Size of the initial design, if not specified, two times the number of HPs is used
    """

    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        initial_design_size: Union[int, None] = None,
    ) -> None:
        if initial_design_size is None:
            initial_design_size = 2 * len(config_space.items())

        self.initial_design_size = initial_design_size
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

        return configs

    def should_propose_config(self, archive: Archive) -> bool:
        return True

    def propose_duel(self, archive: Archive) -> Tuple[int, int]:
        return super().propose_duel(archive)

    def _surrogate_proposal(self, archive: Archive, n: int) -> CS.Configuration:
        """
        Propose a configuration based on a surrogate model.
        """
        x, y = archive.to_torch()
        gp = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ucb = qUpperConfidenceBound(gp, beta=0.1)
        bounds = get_botorch_bounds(self.config_space)
        candidates, _ = optimize_acqf(
            ucb,
            bounds=bounds,
            q=n,
            num_restarts=5,
            raw_samples=20,
        )

        configurations = []
        hp_names = self.config_space.get_hyperparameter_names()

        candidates = [candidates] if n == 1 else candidates.split(1)

        for candidate in candidates:
            hp_values = candidate[0].tolist()

            config_dict = {}

            # candidate contains only floats, round integer HPs
            for hp, val in zip(hp_names, hp_values):
                if not self.config_space.get_hyperparameter(hp).is_legal(val):
                    val = round(val)
                config_dict[hp] = val
            configurations.append(CS.Configuration(self.config_space, config_dict))

        return configurations
