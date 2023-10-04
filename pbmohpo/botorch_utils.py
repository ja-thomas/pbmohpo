from typing import Optional

import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor


# Surrogate Model
# In the long term the whole class should be restructured to be more similar to PairwiseGP
# https://github.com/pytorch/botorch/blob/fa51038c6987174e57ae97138f78287a53a3c6b3/botorch/models/pairwise_gp.py
class VariationalPreferentialGP(GPyTorchModel, ApproximateGP):
    def __init__(
        self,
        train_X: Tensor,
        train_y: Tensor,
        bounds: Tensor,
        use_withening: bool = True,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        """
        Parameters
        ----------
        train_X: torch.Tensor
            A `n x d` tensor of training inputs.

        train_y: torch.Tensor
            A `n x 2` tensor of training outputs.

        bounds: torch.Tensor
            A `2 x d` tensor of lower and upper bounds for each dimension of the
            input space.

        use_withening: bool
            If true, use withening to enhance variational inference.

        covar_module:
            The module computing the covariance matrix.
        """
        input_transform = Normalize(train_X.shape[-1], bounds=bounds)
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        queries, responses = _convert_torch_archive_for_variational_preferential_gp(
            transformed_X, train_y
        )
        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.q = queries.shape[-2]
        self.num_data = queries.shape[-3]
        train_X = queries.reshape(
            queries.shape[0] * queries.shape[1], queries.shape[2]
        )  # Reshape queries in the form of "standard training inputs"
        train_y = responses.squeeze(-1)  # Squeeze out output dimension
        bounds = torch.tensor(
            [[0, 1] for _ in range(self.input_dim)], dtype=torch.float
        ).T  # This assumes the input space has been normalized beforehand
        # Construct variational distribution and strategy
        if use_withening:
            inducing_points = draw_sobol_samples(
                bounds=bounds,
                n=2 * self.input_dim,
                q=1,
                seed=0,
            ).squeeze(1)
            inducing_points = torch.cat([inducing_points, train_X], dim=0)
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
            inducing_points = train_X
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
        # default is the same as the SingleTaskGP covar_module
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=GammaPrior(
                        3.0, 6.0
                    ),  # prior must be 1D https://github.com/cornellius-gp/gpytorch/issues/1317 but scales are always 1 due to normalization so it is fine
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        else:
            self.covar_module = covar_module
        self.train_inputs = (train_X,)
        self.train_targets = train_y
        self.input_transform = input_transform

    # this is only needed for the standard input transformation workflow to pass
    # setting the training data is already done during initialization correctly
    # therefore no transformation of inputs is done during forward if in training mode
    # input transformation in forward mode during eval mode is done correctly through the posterior method of
    # GPyTorchModel base class
    def set_train_data(
        self,
        datapoints: Optional[Tensor] = None,
        comparisons: Optional[Tensor] = None,
        strict: bool = False,
        update_model: bool = True,
    ) -> None:
        None

    def forward(self, X: Tensor) -> MultivariateNormal:
        # https://github.com/pytorch/botorch/blob/cd5c51e6943035c7a1deca3463e1b99f458b9168/botorch/models/gpytorch.py#L367
        # For input transformation usually something like
        # if self.training:
        #     X = self.transform_inputs(X)
        # would be expected but this is already done during initialization for the training data
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
        self.noise = torch.tensor(
            1e-4
        )  # This is only used to draw RFFs-based samples. We set it close to zero because we want noise-free samples
        self.sampler = SobolQMCNormalSampler(
            sample_shape=512
        )  # This allows for SAA-based optimization of the ELBO

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
    (2) evaluating the maximum objective value across the q points for each sample
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
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction` more details.
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


def _convert_torch_archive_for_variational_preferential_gp(
    X: torch.Tensor, y: torch.Tensor
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
    # the comparisons in the archive are in the form of [winner, loser]
    helper_list_X = [torch.stack([X[int(duel[0])], X[int(duel[1])]]) for duel in y]

    new_y = torch.zeros(len(y), dtype=torch.float32).unsqueeze(1)

    new_X = torch.stack(helper_list_X).float()

    return new_X, new_y
