from copy import deepcopy
from typing import Union, Any

import gpytorch
import numpy as np
import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model

from ves.half_ves import HalfVES
from ves.util import optimize_posterior_samples, robust_draw_matheron_paths


class VariationalEntropySearchGammaSeqK(MCAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, torch.Tensor],
            num_paths: int,
            clamp_min: float,
            k: Union[float, torch.Tensor],
            bounds: torch.Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
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
        self.model = deepcopy(model)
        self.best_f = best_f
        self.paths = robust_draw_matheron_paths(self.model, torch.Size([num_paths]))
        self.optimal_outputs = optimize_posterior_samples(
            self.paths,
            bounds,
            device=device
        )
        self.k = k
        self.bounds = bounds
        self.clamp_min = clamp_min

    def forward(
            self,
            x: torch.Tensor,
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

        cur_X = x
        halfVES = HalfVES(
            self.model,
            self.best_f,
            self.paths,
            self.optimal_outputs,
            self.k,
            1.0, # Enforce beta value to be 1.0
            self.clamp_min
        )
        return halfVES(cur_X)
