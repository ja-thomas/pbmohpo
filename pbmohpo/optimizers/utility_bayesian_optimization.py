from pbmohpo.optimizers.optimizer import Optimizer
import ConfigSpace as CS
from typing import List, Union, Dict
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from pbmohpo.archive import Archive
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from pbmohpo.utils import get_botorch_bounds


class UtilityBayesianOptimization(Optimizer):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        initial_design_size: Union[int, None] = None,
    ) -> None:

        if initial_design_size is None:
            initial_design_size = 2 * len(config_space.items())

        self.initial_design_size = initial_design_size
        super().__init__(config_space)

    def propose(self, archive: List) -> Dict:
        if len(archive.data) <= self.initial_design_size:
            return self.confic_space.sample_configuration()
        else:
            return self._surrogate_proposal(archive)

    def _surrogate_proposal(self, archive: Archive) -> CS.Configuration:
        x, y = archive.to_torch()
        gp = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ucb = UpperConfidenceBound(gp, beta=0.1)
        bounds = get_botorch_bounds(self.confic_space)
        candidate, _ = optimize_acqf(
            ucb,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        res = dict(
            zip(self.confic_space.get_hyperparameter_names(), candidate[0].tolist())
        )

        return CS.Configuration(self.confic_space, res)
