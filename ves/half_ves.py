from typing import Union

import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model


class HalfVES(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, torch.Tensor],
        paths,
        optimal_outputs: torch.Tensor,
        k: Union[float, torch.Tensor],
        beta: Union[float, torch.Tensor],
        clamp_min: float = 1e-10,
    ):
        """
        HalfVESGamma is initialized with following args

        Args:
            Model: Gaussian process model
            best_f: y^*_t maximal observation
            paths: Pre-sampled Matheron paths
            optimal_outputs: y^* from the pre-sampled paths
            k: fixed value for k
            beta: fixed value for beta
            clamp_min: minimum value for clamping
        """
        super().__init__(model=model, X_pending=None)
        # Assign values of k and beta
        self.beta = beta
        self.k = k
        # Assign posterior path
        self.paths = paths
        # Assign best value so far y^*_t
        self.best_f = best_f
        # Assign optimal outputs y^*
        self.optimal_outputs = optimal_outputs
        self.clamp_min = clamp_min

    def forward(self, x: torch.Tensor):
        """
        The forward function evaluates ESLB for fixed k and beta
        Follow Eq 3.8
        Args:
            x: batch_size x q=1 x dim input position
        Returns:
            output value
        """
        posterior_samples = self.paths(x.squeeze(1))
        improvement_term = torch.max(posterior_samples, self.best_f).unsqueeze(1)
        # This should be able to be logged, since it is per-sample
        max_value_term = (self.optimal_outputs - improvement_term).clamp_min(
            self.clamp_min
        )
        log_max_value = max_value_term.log()
        improvement_mean = improvement_term.mean(0)
        log_max_mean = log_max_value.mean(0)
        return ((self.k - 1) * log_max_mean + self.beta * improvement_mean).squeeze()
