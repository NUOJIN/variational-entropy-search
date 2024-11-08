from copy import deepcopy
from typing import Union, Any

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.pathwise import MatheronPath

from ves.util import optimize_posterior_samples, robust_draw_matheron_paths, find_root_log_minus_digamma


class HalfVESGamma(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            best_f: Union[float, torch.Tensor],
            paths,
            optimal_outputs: torch.Tensor,
            k: Union[float, torch.Tensor],
            beta: Union[float, torch.Tensor],
            clamp_min: float = 1e-10
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

    def forward(
            self,
            x: torch.Tensor
    ):
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
        max_value_term = (self.optimal_outputs - improvement_term).clamp_min(self.clamp_min)
        log_max_value = max_value_term.log()
        improvement_mean = improvement_term.mean(0)
        log_max_mean = log_max_value.mean(0)
        return ((self.k - 1) * log_max_mean + self.beta * improvement_mean).squeeze()


class VariationalEntropySearchGamma(MCAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, torch.Tensor],
            paths: MatheronPath,
            clamp_min: float,
            optimize_acqf_options: dict[str, Any] | None = None,
            bounds: torch.Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
            stop_tolerance_coeff: float = 1e-5,
            device: torch.device = torch.device("cpu"),
            **kwargs: Any,
    ):
        """
        The VES(-Gamma) class should be initialized with following args

        Args:
            model: Gaussian Process model
            best_f: y_t^* the best observation
            paths: Sampled Matheron paths
            bounds: D x 2 boundary
            clamp_min: minimum value for clamping
            stop_tolerance_coeff: stopping tolerance coefficient
            optimize_acqf_options: options for the optimizer
        """
        super().__init__(model=model)
        self.sampling_model = deepcopy(model)
        self.best_f = best_f
        self.optimal_outputs = optimize_posterior_samples(
            paths,
            bounds,
            device=device
        )
        self.paths = paths
        self.bounds = bounds
        self.clamp_min = clamp_min
        self.stop_tolerance_coeff = stop_tolerance_coeff
        if optimize_acqf_options is None:
            optimize_acqf_options = {
                "num_restarts": 5,
                "raw_samples" : 1024,
                "options"     : {"sample_around_best": True}
            }
        self.optimize_acqf_options = optimize_acqf_options

    def forward(
            self,
            x: torch.Tensor,
            num_paths: int,
            num_iter: int = 64,
            show_progress: bool = True,
            **kwargs: Any,
    ):
        """
        This VES class implements VES-Gamma, a special case of VES.
        There are two steps: the first step is to design a new halfVES class, which
        is uniquely specified by the value of k and beta. A optimize_acqf optimizer
        is implemented to find its optimal candidate. The second step is to use this
        selected optimal candidate to find the next k and beta, and generate a new
        halfVES class. The iteration is repeated num_iter times.

        Args:
            x: The initial value for X batch_size x q x dim
            num_iter: number of iterations
            num_paths: number of paths to sample
            show_progress: whether to show progress
        Returns:
            candidate: the optimal position for the solution
            acq_value: the value of its position
        """
        assert num_iter > 0, "Number of iterations should be positive"

        cur_X = x
        device = cur_X.device
        for i in range(num_iter):
            # Step 1: Find current optimal k and beta
            max_value_term = self.generate_max_value_term(cur_X)
            kval, betaval = self.find_k(max_value_term)
            halfVES = HalfVESGamma(
                self.model,
                self.best_f,
                self.paths,
                self.optimal_outputs,
                kval.item(),
                betaval.item(),
                self.clamp_min
            )
            # Step 2: Given k and beta, find optimal X
            new_x, acq_value = optimize_acqf(
                halfVES,
                bounds=self.bounds.T,
                q=1,  # Number of candidates to optimize for
                **self.optimize_acqf_options
            )
            if torch.norm(new_x - cur_X) < self.stop_tolerance_coeff * self.bounds.size(0):
                break
            cur_X = new_x
            self.paths = robust_draw_matheron_paths(self.model, torch.Size([num_paths]))
            self.optimal_outputs = optimize_posterior_samples(
                self.paths,
                self.bounds,
                device=device,
            )
            if show_progress and i % 5 == 0:
                print(f"Iteration {i}: K: {kval.item():.3e}; beta {betaval.item():.3e}; AF value: {acq_value:.3e}")
        return cur_X, acq_value, kval.item(), betaval.item()

    def generate_max_value_term(
            self,
            x: torch.Tensor
    ):
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
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(self.clamp_min)
        # This should be able to be logged, since it is per-sample
        return max_value_term

    def find_k(
            self,
            max_value_term: torch.Tensor
    ):
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
        return k_vals, beta_vals

    def root_finding(
            self,
            x: torch.Tensor
    ):
        """
        Root finding function to solve Eq 3.9; Non-differentiable(?)
        """
        dtype, device = x.dtype, x.device
        x_np = x.flatten().detach().cpu().numpy()
        res = np.zeros_like(x_np)
        for i, intercept in enumerate(x_np):
            res[i] = find_root_log_minus_digamma(intercept)
        return torch.Tensor(res).reshape(x.shape).to(dtype=dtype, device=device)
