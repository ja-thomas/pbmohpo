from typing import List, Union

import ConfigSpace as CS
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from pbmohpo.archive import UtilityArchive
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

    def propose(self, archive: UtilityArchive) -> CS.Configuration:
        """
        Propose a new configuration to evaluate.

        Takes an list of previous evaluations and proposes a new configuration to evaluate.
        If the number of observations in the archive is smaller than the initial design, propose a random configuration.

        Parameters
        ----------
        archive: List
            List of previous evaluations

        Returns
        -------
        CS.Configuration:
            Proposed Configuration

        """
        if len(archive.data) <= self.initial_design_size:
            return self.config_space.sample_configuration()
        else:
            return self._surrogate_proposal(archive)

    def _surrogate_proposal(self, archive: UtilityArchive) -> CS.Configuration:
        """
        Propose a configuration based on a surrogate model.
        """
        x, y = archive.to_torch()
        gp = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ucb = UpperConfidenceBound(gp, beta=0.1)
        bounds = get_botorch_bounds(self.config_space)
        candidate, _ = optimize_acqf(
            ucb,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        hp_names = self.config_space.get_hyperparameter_names()
        hp_values = candidate[0].tolist()

        config_dict = {}

        # candidate contains only floats, round integer HPs
        for hp, val in zip(hp_names, hp_values):
            if not self.config_space.get_hyperparameter(hp).is_legal(val):
                val = round(val)
            config_dict[hp] = val

        return CS.Configuration(self.config_space, config_dict)
