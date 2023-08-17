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
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls.variational_elbo import VariationalELBO

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

        model = PairwiseGP(x, y)
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        bounds = get_botorch_bounds(self.config_space)
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

    Implements the expected utility of the best option algrithm.

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

        X, _ = archive.to_torch()
        y = torch.Tensor(archive.comparisons)

        X, y = self._convert_torch_archive_for_variational_preferential_gp(X, y)

        model = VariationalPreferentialGP(X, y)
        model.train()
        model.likelihood.train()

        mll = VariationalELBO(
            likelihood=model.likelihood,
            model=model,
            # Magic num proposed in https://github.com/facebookresearch/qEUBO
            num_data=2 * model.num_data,
        )

        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=64)
        acq_func = qExpectedUtilityOfBestOption(model=model, sampler=sampler)

        bounds = get_botorch_bounds(self.config_space)
        bounds = bounds.type(torch.Tensor)

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

    def _convert_torch_archive_for_variational_preferential_gp(
        self, X: torch.Tensor, y: torch.Tensor
    ):
        """
        Converts inputs and targets to the form the implementation of the
        variational preferential gp needs.

        Parameters
        ----------
        X: torch.Tensor
            Configs in archive as given by archive.to_torch()
        y: torch.Tensor
            Recorded duels in archive as given by archive.to_torch()

        Returns
        -------
        new_X: torch.Tensor
            An n x q x n tensor Each of the `n` queries is constituted
            by `q` `d`-dimensional decision vectors.

        new_y: torch.Tensor
            An n x 1 tensor of training outputs. Each of the `n` responses is
            an integer between 0 and `q-1` indicating the decision vector
            selected by the user.
        """
        helper_list_X = []

        for duel in y:
            helper_list_X = [
                torch.stack([X[int(duel[0])], X[int(duel[1])]]) for duel in y
            ]

        new_X = torch.stack(helper_list_X)
        new_y = torch.ones(len(new_X), dtype=torch.float32)

        return new_X.type(torch.Tensor), new_y
