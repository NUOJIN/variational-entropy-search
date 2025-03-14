from typing import Union

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model

from ves.util import (
    find_root_log_minus_digamma,
)


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


class HalfVESNew(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            best_f: Union[float, torch.Tensor],
            paths,
            optimal_outputs: torch.Tensor,
            clamp_min: float = 1e-10,
            reg_lambda: float = 0.0,
            reg_target: float = 1.0,
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
        # Assign posterior path
        self.paths = paths
        # Assign best value so far y^*_t

        self.best_f = best_f
        # Assign optimal outputs y^*
        self.optimal_outputs = optimal_outputs
        self.clamp_min = clamp_min
        self.beta_val = None
        self.k_val = None
        self.reg_lambda = reg_lambda
        self.reg_target = reg_target

    def forward(self, x: torch.Tensor):
        """
        The forward function evaluates ESLB for fixed k and beta
        Follow Eq 3.8
        Args:
            x: batch_size x q=1 x dim input position
        Returns:
            output value
        """
        max_value_term, improvement_term = self.generate_max_value_term(x)
        kval, betaval = self.find_k(max_value_term)
        self.beta_val = betaval
        self.k_val = kval
        log_max_value = max_value_term.log()
        improvement_mean = improvement_term.mean(0)
        log_max_mean = log_max_value.mean(0)
        return ((kval - 1) * log_max_mean + betaval * improvement_mean).squeeze()

    def generate_max_value_term(self, x: torch.Tensor):
        """
        This function generate values of y^* - max(y_x, y^*_t) given
        position X and paths.
        Args:
            x: current inputs. Size: batch_size x q=1 x dim
        Return:
            max_value_term: y^* - max(y_x, y^*_t).
            Size: NUM_PATH x batch_size
        """
        posterior_samples = self.paths(x.squeeze(1))
        improvement_term = torch.max(posterior_samples, self.best_f)
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(
            self.clamp_min
        )
        # This should be able to be logged, since it is per-sample
        return max_value_term, improvement_term.unsqueeze(1)

    def find_k(self, max_value_term: torch.Tensor):
        """
        This function evaluates the optimal values of k and beta
        Args:
            max_value_term: NUM_PATH x q x batch_size
        Return:
            k_vals: q x batch_size
            beta_vals: q x batch_size
        """
        A = max_value_term.mean(dim=0)
        B = (torch.log(max_value_term)).mean(dim=0)
        self.v = torch.log(A) - B
        k_vals = self.root_finding(self.v)
        beta_vals = k_vals / A
        return k_vals.detach(), beta_vals.detach()

    def root_finding(self, x: torch.Tensor):
        """
        Root finding function to solve Eq 3.9; Non-differentiable(?)
        """
        dtype, device = x.dtype, x.device
        x_np = x.flatten().detach().cpu().numpy()
        res = np.zeros_like(x_np)
        for i, intercept in enumerate(x_np):
            res[i] = find_root_log_minus_digamma(intercept, reg_lambda=self.reg_lambda, reg_target=self.reg_target)
        return torch.Tensor(res).reshape(x.shape).to(dtype=dtype, device=device)
