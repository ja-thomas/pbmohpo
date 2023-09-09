import logging
from itertools import combinations
from math import comb
from typing import List, Optional, Tuple

import ConfigSpace as CS
import numpy as np
import torch
from botorch.acquisition import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior

from pbmohpo.archive import Archive
from pbmohpo.botorch_utils import (
    VariationalPreferentialGP,
    qExpectedUtilityOfBestOption,
)
from pbmohpo.optimizers.optimizer import BayesianOptimization
from pbmohpo.utils import get_botorch_bounds


class EUBO(BayesianOptimization):
    """
    Bayesian Optimization for Pairwise Comparison Data.

    Implements the Analytic Expected Utility Of Best Option algrithm.
    For details see: https://botorch.org/tutorials/preference_bo

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
            initial_design_size = 4 * len(config_space.items())

        self.initial_design_size = initial_design_size
        self.new_configs = 0
        super().__init__(config_space)

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
            List of tuples of two indices of Archive evaluations to compare

        """

        evals = len(archive.evaluations)

        if len(archive.comparisons) == 0:
            n = min(
                self.initial_design_size * n,
                comb(self.initial_design_size, 2),
            )
            logging.info(f"Running: Initial duels of size {n}")
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

        x, _ = archive.to_torch()
        y = torch.Tensor(archive.comparisons)
        bounds = get_botorch_bounds(self.config_space)

        # covar_module mostly taken from PairwiseGP but using a Matern 2.5 Kernel instead of RBF
        # somewhat similar to the SingleTaskGP covar_module
        os_lb, os_ub = 1e-2, 1e2
        ls_prior = GammaPrior(2.4, 2.7)
        ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=x.shape[-1],
                lengthscale_prior=ls_prior,
                lengthscale_constraint=GreaterThan(
                    lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
                ),
                dtype=torch.float64,
            ),
            outputscale_prior=SmoothedBoxPrior(a=os_lb, b=os_ub),
            # Make sure we won't get extreme values for the output scale
            outputscale_constraint=Interval(
                lower_bound=os_lb * 0.5,
                upper_bound=os_ub * 2.0,
                initial_value=1.0,
            ),
            dtype=torch.float64,
        )

        model = PairwiseGP(
            x,
            y,
            covar_module=covar_module,
            input_transform=Normalize(x.shape[-1], bounds=bounds),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        candidates, acq_val = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=n,  # FIXME: this will always fail if n is not 2 or 1 with a previous winner specified which we do not do?
            num_restarts=10,
            raw_samples=1000,
        )

        logging.debug(f"Acquisition function value: {acq_val}")

        configs = self._candidates_to_configs(candidates, n)
        return configs


class qEUBO(EUBO):
    """
    Bayesian Optimization for Pairwise Comparison Data.

    Implements the expected utility of the best option algorithm.

    For details see: https://arxiv.org/pdf/2303.15746.pdf

    Their implementation can be found here: https://github.com/facebookresearch/qEUBO

    Large parts of that code have been used in this implementation.

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
        super().__init__(config_space, initial_design_size)

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

        x, _ = archive.to_torch()
        y = torch.Tensor(archive.comparisons)
        bounds = get_botorch_bounds(self.config_space).float()
        model = VariationalPreferentialGP(x, y, bounds=bounds)
        model.train()
        model.likelihood.train()

        mll = VariationalELBO(
            likelihood=model.likelihood,
            model=model,
            # https://github.com/facebookresearch/qEUBO/blob/21cd661efc25b242c9fdf5230f5828f01ff0872b/src/utils.py#L36
            num_data=2 * model.num_data,
        )

        fit_gpytorch_mll(
            mll,
            optimizer=fit_gpytorch_mll_scipy,
            optimizer_kwargs={"method": "L-BFGS-B", "options": {"maxls": 100}},
        )  # this is passed on to fit_gpytorch_mll_scipy

        sampler = SobolQMCNormalSampler(sample_shape=64)
        acq_func = qExpectedUtilityOfBestOption(model=model, sampler=sampler)

        candidates, acq_val = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=n,
            num_restarts=10,
            raw_samples=1000,
        )

        logging.debug(f"Acquisition function value: {acq_val}")

        configs = self._candidates_to_configs(candidates, n)
        return configs
