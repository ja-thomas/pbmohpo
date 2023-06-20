from typing import Optional

import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)

# from pbmohpo.models.likelihoods.preferential_softmax_likelihood import \
# PreferentialSoftmaxLikelihood
from torch import Tensor


# Surrogate Model
class VariationalPreferentialGP(GPyTorchModel, ApproximateGP):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        use_withening: bool = True,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        """
        Parameters
        ----------
        queries: torch.Tensor
            A `n x q x d` tensor of training inputs. Each of the `n` queries
            is constituted by `q` `d`-dimensional decision vectors.

        responses: torch.Tensor
            A `n x 1` tensor of training outputs. Each of the `n` responses is
            an integer between 0 and `q-1` indicating the decision vector
            selected by the user.

        use_withening: bool
            If true, use withening to enhance variational inference.

        covar_module:
            The module computing the covariance matrix.
        """
        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.q = queries.shape[-2]
        self.num_data = queries.shape[-3]
        train_x = queries.reshape(
            queries.shape[0] * queries.shape[1], queries.shape[2]
        )  # Reshape queries in the form of "standard training inputs"
        train_y = responses.squeeze(-1)  # Squeeze out output dimension
        bounds = torch.tensor(
            [[0, 1] for _ in range(self.input_dim)],
            dtype=torch.float
            # [[0, 1] for _ in range(self.input_dim)], dtype=torch.double
        ).T  # This assumes the input space has been normalized beforehand
        # Construct variational distribution and strategy
        if use_withening:
            inducing_points = draw_sobol_samples(
                bounds=bounds,
                n=2 * self.input_dim,
                q=1,
                seed=0,
            ).squeeze(1)
            inducing_points = torch.cat([inducing_points, train_x], dim=0)
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        else:
            inducing_points = train_x
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        super().__init__(variational_strategy)
        self.likelihood = PreferentialSoftmaxLikelihood(num_alternatives=self.q)
        self.mean_module = ConstantMean()
        scales = bounds[1, :] - bounds[0, :]

        if covar_module is None:
            self.covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0 / scales),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        else:
            self.covar_module = covar_module
        self._num_outputs = 1
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X)

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return 1


# Likelihood
class PreferentialSoftmaxLikelihood(Likelihood):
    """
    Implements the softmax likelihood used for GP-based preference learning.

    Underlying math:
    p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    Parameters
    ----------
    num_alternatives: int
        Number of alternatives (i.e., q)
    """

    def __init__(self, num_alternatives):
        super().__init__()
        self.num_alternatives = num_alternatives
        self.noise = torch.tensor(1e-4)  # This is only used to draw RFFs-based
        # samples. We set it close to zero because we want noise-free samples
        self.sampler = SobolQMCNormalSampler(sample_shape=512)  # This allows
        # for SAA-based optimization of the ELBO

    def _draw_likelihood_samples(
        self, function_dist, *args, sample_shape=None, **kwargs
    ):
        function_samples = self.sampler(GPyTorchPosterior(function_dist)).squeeze(-1)
        return self.forward(function_samples, *args, **kwargs)

    def forward(self, function_samples, *params, **kwargs):
        function_samples = function_samples.reshape(
            function_samples.shape[:-1]
            + torch.Size(
                (
                    int(function_samples.shape[-1] / self.num_alternatives),
                    self.num_alternatives,
                )
            )
        )  # Reshape samples as if they came from a multi-output model (with `q` outputs)
        num_alternatives = function_samples.shape[-1]

        if num_alternatives != self.num_alternatives:
            raise RuntimeError("There should be %d points" % self.num_alternatives)

        res = base_distributions.Categorical(logits=function_samples)  # Passing the
        # function values as logits recovers the softmax likelihood
        return res


# Acquisition Function
class qExpectedUtilityOfBestOption(MCAcquisitionFunction):
    r"""Expected Utility of Best Option (qEUBO).

    This computes qEUBO by
    (1) sampling the joint posterior over q points
    (2) evaluating the maximum objective value accross the q points for each sample
    (3) averaging over the samples

    `qEUBO(X) = E[max Y], Y ~ f(X), where X = (x_1,...,x_q)`
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_baseline: Optional[Tensor] = None,
    ) -> None:
        r"""MC-based Expected Utility of the Best Option (qEUBO).

        Args:
            model: A fitted model.
             sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            X_baseline:  A `m x d`-dim Tensor of `m` design points forced to be included
                in the query (in addition to the q points, so the query is constituted
                by q + m alternatives). Concatenated into X upon forward call. Copied and
                set to have no gradient. This is useful, for example, if we want to force
                one of the alternatives to be the point chosen by the decision-maker in
                the previous iteration.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            X_pending=X_baseline,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qEUBO on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of qEUBO values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        posterior_X = self.model.posterior(X)
        Y_samples = self.sampler(posterior_X)
        util_val_samples = self.objective(Y_samples)
        best_util_val_samples = util_val_samples.max(dim=-1).values
        exp_best_util_val = best_util_val_samples.mean(dim=0)
        return exp_best_util_val
