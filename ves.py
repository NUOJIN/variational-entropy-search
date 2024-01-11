from __future__ import annotations
from copy import deepcopy
from typing import Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling import SobolEngine
from botorch.sampling.pathwise import MatheronPath
from torch import Tensor
from torch.special import digamma, polygamma

NUM_OPTIMA = 100

def plot_ves(ves, train_X=None, train_Y=None, f=None):
    '''
    Plotting function for visualizing the VES acquisition function in 1D case
    '''
    def dt(d): return d.detach().numpy()
    X = torch.linspace(0, 1, 201).unsqueeze(-1).unsqueeze(-1).to(torch.double)
    X = torch.linspace(0, 1, 201).unsqueeze(-1).to(torch.double)

    fig, axes = plt.subplots(3, 1, figsize=(20, 10))

    posterior = ves.model.posterior(X)
    m = dt(posterior.mean)
    s = dt(posterior.variance.sqrt())
    axes[0].plot(dt(X).flatten(), m.flatten(), label='posterior mean')
    if train_X is not None and f is not None:
        axes[0].plot(dt(X).flatten(), f(X).flatten(), color='black', label='true function')
        axes[0].scatter(dt(train_X).flatten(), dt(train_Y).flatten())
    else:
        axes[0].scatter(dt(ves.model.train_inputs[0]), dt(ves.model.train_targets))
    axes[0].fill_between(dt(X).flatten(), (m - 2 * s).flatten(), (m + 2 * s).flatten(), alpha=0.2)
    axes[0].legend()

    kvals, betavals, ei, val = ves(X, return_ves=True)
    axes[1].plot(X.flatten(), dt(kvals.flatten()), label=f'k', color='navy')
    axes[1].plot(X.flatten(), dt(betavals.flatten()), label=f'beta', color='green')
    val = dt(val.flatten())
    val = (val - np.mean(val))/np.std(val)
    ei = dt(ei.flatten())
    ei = (ei - np.mean(ei))/np.std(ei)
    axes[2].plot(X.flatten(), val, label=f'VES'.upper(), color='red')
    axes[2].plot(X.flatten(), ei, label=f'EI'.upper(), color='blue')
    axes[2].axvline(X.flatten()[val.argmax()], label=f'__nolabel__', color='red')
    axes[2].axvline(X.flatten()[ei.argmax()], label=f'__nolabel__', color='blue')
    axes[2].set_title('VES and EI Function', fontsize=18)

    plt.tight_layout()
    axes[1].legend()
    axes[2].legend()
    return fig

def optimize_posterior_samples(
    paths,
    bounds: Tensor,
    maximize: Optional[bool] = True,
    candidates: Optional[Tensor] = None,
    raw_samples: Optional[int] = 2048,
    num_restarts: Optional[int] = 10,
    maxiter: int = 100,
    spray_points: int = 20,
    lr: float = 2.5e-4
) -> Tuple[Tensor, Tensor]:
    r"""Cheaply optimizes posterior samples by random querying followed by vanilla
    gradient descent on the best num_restarts points.

    Args:
        paths: Tample paths from the GP which can be called via forward()
        x: evaluation position for y_x
        bounds: The bounds on the search space.
        maximize: Whether or not to maximize or minimize the posterior samples.
        candidates: A priori good candidates (typically previous design points)
            which acts as extra initial guesses for the optimization routine.
        raw_samples: The number of samples with which to query the samples initially.
        num_restarts The number of gradient descent steps to use on each of the best 
        found initial points.
        maxiter: The maximal permitted number of gradient descent steps.
        lr: The stepsize of the gradient descent steps.

    Returns:Optional
        Tuple[Tensor, Tensor]: The optimal input-output pair(s) (X^*. f^*)
    """
    candidate_set = SobolEngine(
        dimension=bounds.shape[0], scramble=True).draw(raw_samples)
    # TODO add spray points
    # queries all samples on all candidates - output raw_samples * num_objectives * num_optima
    candidate_queries = paths.forward(candidate_set)
    num_optima = candidate_queries.shape[0]
    batch_size = candidate_queries.shape[1] if candidate_queries.ndim == 3 else 1 
    argtop_candidates = candidate_queries.argsort(dim=-1, descending=True)[
        ..., 0:num_restarts]
    
    # These are used as masks when retrieving the argmaxes
    row_indexer = torch.arange(num_optima * batch_size).to(torch.long)
    X_argtop = candidate_set[argtop_candidates, :].requires_grad_(requires_grad=True)
    for i in range(maxiter):
        per_sample_outputs = paths.forward(X_argtop)
        grads = torch.autograd.grad(
            per_sample_outputs, X_argtop, grad_outputs=torch.ones_like(per_sample_outputs))[0]
        X_argtop = torch.clamp(X_argtop + lr * grads, 0, 1)  # TODO fix bounds here
    
    per_sample_outputs = paths.forward(X_argtop).reshape(num_optima * batch_size, num_restarts)
    f_max = per_sample_outputs.max(axis=-1).values.reshape(num_optima, batch_size, 1)
    
    return f_max.detach()

def find_root_log_minus_digamma(xx, initial_guess, tol=1e-5, max_iter=100):
    """
    Find a root of the function log(x) - digamma(x) - xx using Newton's method.

    Args:
    xx (float or tensor): The constant value to subtract in the function.
    initial_guess (float or tensor): Initial guess for the root.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.

    Returns:
    float or tensor: Approximate root of the function.
    """
    x = initial_guess
    for _ in range(max_iter):
        value = torch.log(x) - digamma(x) - xx
        derivative = 1/x - polygamma(1, x)  # derivative of the function

        # Newton's method update
        x_new = x - value / derivative

        # Check for convergence
        if torch.abs(x_new - x) < tol:
            return x_new

        x = x_new

    return torch.tensor(1.0)

class VariationalEntropySearch(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        paths: MatheronPath,
        X_pending: Optional[Tensor] = None,
        bounds: Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
        maximize: bool = True,
        plot: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model=model, X_pending=X_pending)
        self.sampling_model = deepcopy(model)
        self.best_f = best_f
        self.optimal_outputs = optimize_posterior_samples(
            paths, 
            bounds)
        self.paths = paths

        if plot and model.train_inputs[0].shape[-1] == 1:
            plot_ves(self)

    def forward(self, X, beta: float = 1.0, k: float = 1.0, return_ves = False):
        posterior_samples = self.paths.forward(X)
        improvement_term = torch.max(posterior_samples, self.best_f)
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(1e-3) # This should be able to be logged, since it is per-sample
        kvals, betavals = self.find_k(max_value_term)
        reg_term_new = kvals * torch.log(betavals) - torch.lgamma(kvals)
        reg_term = k * torch.log(Tensor([beta])) - torch.lgamma(Tensor([k]))
        if return_ves: 
            return kvals, betavals, -max_value_term.mean(dim=0).squeeze(-1), ((reg_term_new + (kvals - 1) * torch.log(max_value_term) - betavals * max_value_term)).mean(dim=0).squeeze(-1).squeeze(-1)

        return ((reg_term + (k - 1) * torch.log(max_value_term) - beta * max_value_term)).mean(dim=0).squeeze(-1).squeeze(-1)
    
    def find_k(self, max_value_term: Tensor):
        A = max_value_term.mean(dim=0)
        B = (torch.log(max_value_term)).nanmean(dim=0)
        self.v = torch.log(A) - B
        k_vals = self.root_finding(self.v)
        beta_vals = k_vals / A
        return k_vals, beta_vals

    def root_finding(self, x: Tensor):
        res = np.zeros_like(x.flatten().detach().numpy())
        for i, xx in enumerate(x.flatten().detach().numpy()):
            res[i] = find_root_log_minus_digamma(torch.tensor(xx), initial_guess=torch.tensor(0.5))
        return torch.Tensor(res).reshape(x.shape)
