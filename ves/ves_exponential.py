from copy import deepcopy
from typing import Union, Any

import gpytorch
import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.optim import optimize_acqf

from ves.half_ves import HalfVES
from ves.util import optimize_posterior_samples


class VariationalEntropySearchExponential(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, torch.Tensor],
        paths,
        clamp_min: float,
        optimize_acqf_options: dict[str, Any] | None = None,
        bounds: torch.Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
        device: torch.device = torch.device("cpu"),
        **kwargs: Any,
    ):
        """
        The VES(-Gamma) class should be initialized with following args
        model: Gaussian Process model
        best_f: y_t^* the best observation
        paths: Sampled Matheron paths
        bounds: D x 2 boundary
        """
        super().__init__(model=model)
        self.sampling_model = deepcopy(model)
        self.best_f = best_f
        self.optimal_outputs = optimize_posterior_samples(paths, bounds, device=device)
        self.paths = paths
        self.bounds = bounds
        self.clamp_min = clamp_min
        if optimize_acqf_options is None:
            optimize_acqf_options = {
                "num_restarts": 5,
                "raw_samples": 1024,
                "options": {"sample_around_best": True},
            }
        self.optimize_acqf_options = optimize_acqf_options

    def forward(
        self,
        x: torch.Tensor,
        num_iter: int = 64,
        **kwargs: Any,
    ):
        """
        This VES class implements VES-Exp, a special case of VES, and expected to be
        equivalent to EI.
        Args:
            x: The initial value for X batch_size x q x dim
            num_iter: number of iterations
        Returns:
            candidate: the optimal position for the solution
            acq_value: the value of its position
        """
        assert num_iter > 0, "Number of iterations should be positive"

        current_x = x
        # solve beta (or lambda)
        betaval = torch.ones(current_x.shape[0])
        kval = torch.ones_like(betaval)
        halfVES = HalfVES(
            self.model,
            self.best_f,
            self.paths,
            self.optimal_outputs,
            kval.item(),
            betaval.item(),
            self.clamp_min,
        )
        # Step 2: Given k and beta, find optimal X
        with gpytorch.settings.cholesky_max_tries(9):
            current_x, acq_value = optimize_acqf(
                halfVES,
                bounds=self.bounds.T,
                q=1,  # Number of candidates to optimize for
                **self.optimize_acqf_options,
            )

        return current_x, acq_value, kval.item(), betaval.item()

    def generate_max_value_term(self, X: torch.Tensor):
        """
        This function generate values of y^* - max(y_x, y^*_t) given
        position X and paths.
        Args:
            X: current inputs. Size: batch_size x q=1 x dim
        Return:
            max_value_term: y^* - max(y_x, y^*_t).
            Size: NUM_PATH x batch_size
        """
        posterior_samples = self.paths(X.squeeze(1))
        improvement_term = torch.max(posterior_samples, self.best_f)
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(
            self.clamp_min
        )
        # This should be able to be logged, since it is per-sample
        return max_value_term
