from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from botorch import settings

from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from botorch.sampling.pathwise import MatheronPath
from botorch.utils.plot_acq import plot_ves

from torch import Tensor

EXP = Tensor([0.1])

class VariationalEntropySearch(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        paths: MatheronPath,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        maximize: bool = True,
        plot: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model=model, X_pending=X_pending)
        self.sampling_model = deepcopy(model)
        self.sampling_model.set_paths(paths)
        self.best_f = best_f
        self.optimal_inputs = optimal_inputs.unsqueeze(-2)
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)

        if plot and model.train_inputs[0].shape[-1] == 1:
            plot_ves(self)

    #@concatenate_pending_points
    # @t_batch_mode_transform()
    def forward(self, X, beta: float = EXP, k: float = EXP, split = False):
        reg_term = k * torch.log(beta) - torch.lgamma(k)
        posterior_samples = self.sampling_model.posterior(X).rsample()
        improvement_term = torch.max(posterior_samples, self.best_f)
        max_value_term = (self.optimal_outputs - improvement_term).clamp_min(1e-3) # This should be able to be logged, since it is per-sample
        
        if split:
            return (
                reg_term, 
                (k - 1) * torch.log(max_value_term).mean(dim=0).squeeze(-1), 
                -(beta * max_value_term).mean(dim=0).squeeze(-1)
            )

        return ((reg_term + (k - 1) * torch.log(max_value_term) - beta * max_value_term)).mean(dim=0).squeeze(-1)
