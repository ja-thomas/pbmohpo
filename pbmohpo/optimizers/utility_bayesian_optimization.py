import logging
from typing import List, Optional, Tuple

import ConfigSpace as CS
from botorch.acquisition import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from pbmohpo.archive import Archive
from pbmohpo.optimizers.optimizer import BayesianOptimization
from pbmohpo.utils import get_botorch_bounds


class UtilityBayesianOptimization(BayesianOptimization):
    """
    Single objective Bayesian optimization of utility scores.

    Implements a simple BO loop to optimize the utility scores provided by the decision maker.
    Uses a GP surrogate and UCB acquisition function with beta=0.1.

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
        initial_design_size: Optional[int] = None,
    ) -> None:
        if initial_design_size is None:
            initial_design_size = 2 * len(config_space.items())

        self.initial_design_size = initial_design_size
        super().__init__(config_space)

    def propose_duel(self, archive: Archive, n: int = 1) -> List[Tuple[int, int]]:
        return super().propose_duel(archive, n)

    @property
    def dueling(self) -> bool:
        return False

    def _surrogate_proposal(self, archive: Archive, n: int) -> List[CS.Configuration]:
        """
        Propose a new configuration by surrogate

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

        x, y = archive.to_torch()
        gp = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        fit_gpytorch_mll(mll)
        ucb = qUpperConfidenceBound(gp, beta=0.1)
        bounds = get_botorch_bounds(self.config_space)
        candidates, acq_val = optimize_acqf(
            ucb,
            bounds=bounds,
            q=n,
            num_restarts=5,
            raw_samples=20,
        )

        logging.debug(f"Acquisition function value: {acq_val}")

        configs = self._candidates_to_configs(candidates, n)
        return configs
