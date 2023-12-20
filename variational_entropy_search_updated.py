# %%
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
import scipy
from torch import Tensor

# %%
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling import SobolEngine
num_optima = 100

def plot_ves(ves, betas=[1, 2, 4], ks=[0.1, 1.0, 10.0], train_X=None, train_Y=None, f=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    c = ['navy', 'forestgreen', 'brown']
    alpha = [0.3, 0.7, 1]
    for b_idx, beta in enumerate(betas):
        for g_idx, k in enumerate(ks):
            tensorized_params = (Tensor([beta]), Tensor([k]))
            res = dt(ves(X, beta=beta, k=k))
            axes[1].plot(
                X.flatten(), res, label=f'Beta: {beta}, k: {k}', alpha=alpha[g_idx], color=c[b_idx])
            axes[1].axvline(
                X.flatten()[res.argmax()], label=f'__nolabel__', alpha=alpha[g_idx], color=c[b_idx])

    ei, val = ves(X, return_ves=True)
    # axes[2].plot(X.flatten(), dt(reg.flatten() * torch.ones_like(X.flatten())), label=f'Reg', color='purple')
    # axes[2].plot(X.flatten(), dt(ei).flatten(), label=f'ei'.upper(), color='orange')
    # axes[2].plot(X.flatten(), dt(mv).flatten(), label=f'mv'.upper(), color='teal')
    # axes[2].plot(X.flatten(), dt(reg + ei + mv).flatten(), label=f'Total', color='black')
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
    #plt.savefig(f'fig_iter{len(prior_acq.sampling_model.train_targets)}.pdf')
    plt.show()

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
    # sample_argmax = torch.max(per_sample_outputs, dim=-1, keepdims=True)[1].flatten()
    # X_argtop_flat = X_argtop.reshape(num_optima * batch_size, num_restarts, -1)
    
    # X_max = X_argtop_flat[row_indexer, sample_argmax].reshape(num_optima, batch_size, -1)
    f_max = per_sample_outputs.max(axis=-1).values.reshape(num_optima, batch_size, 1)
    
    return f_max.detach()

# %%
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
        # self.sampling_model.set_paths(paths)
        self.best_f = best_f
        
        self.optimal_inputs = optimal_inputs.unsqueeze(-2)
        self.optimal_outputs = optimize_posterior_samples(
            paths, 
            torch.Tensor([[torch.zeros(1), torch.ones(1)]]))
        self.paths = paths

        if plot and model.train_inputs[0].shape[-1] == 1:
            plot_ves(self)

    #@concatenate_pending_points
    # @t_batch_mode_transform()
    def forward(self, X, beta: float = EXP, k: float = EXP, return_ves = False):
        # posterior_samples = self.sampling_model.posterior(X).rsample(self.paths)
        posterior_samples = self.paths.forward(X)
        improvement_term = torch.max(posterior_samples, self.best_f)
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(1e-3) # This should be able to be logged, since it is per-sample
        kvals, betavals = self.find_k(max_value_term)
        print(kvals.shape, betavals.shape)
        reg_term_new = kvals * torch.log(betavals) - torch.lgamma(kvals)
        reg_term = k * torch.log(Tensor([beta])) - torch.lgamma(Tensor([k]))
        if return_ves: # won't need values of k and beta
            return -max_value_term.mean(dim=0).squeeze(-1), ((reg_term_new + (kvals - 1) * torch.log(max_value_term) - betavals * max_value_term)).mean(dim=0).squeeze(-1).squeeze(-1)
            # return (
            #     reg_term, 
            #     (k - 1) * torch.log(max_value_term).mean(dim=0).squeeze(-1), 
            #     -(beta * max_value_term).mean(dim=0).squeeze(-1)
            # )
        return ((reg_term + (k - 1) * torch.log(max_value_term) - beta * max_value_term)).mean(dim=0).squeeze(-1).squeeze(-1)
    
    def find_k(self, max_value_term: Tensor):
        A = max_value_term.mean(dim=0)
        B = (torch.log(max_value_term[max_value_term != 0.0])).nanmean(dim=0)
        k_vals = self.root_finding(torch.log(A) - B)
        beta_vals = k_vals / A
        return k_vals, beta_vals

    def root_finding(self, x: Tensor):
        res = np.zeros_like(x.flatten().detach().numpy())
        for i, xx in enumerate(x.flatten().detach().numpy()):
            func = lambda x: np.log(x) - scipy.special.digamma(x) - xx
            kx = scipy.optimize.root_scalar(func, bracket=[0.0,100.0], method='bisect').root
            if np.isnan(kx) or kx == np.inf or kx == -np.inf:
                kx = 1.0
            res[i] = kx
        return torch.Tensor(res).reshape(x.shape)
        

# %%
import numpy as np
### test VES 1D
## design training data
train_X = torch.rand(2, 1, dtype=torch.double)
def f(X,noise=0.0):
    return -np.sin(15*X)*X**2
train_Y = f(train_X,noise=0.0)
bounds = torch.Tensor([[torch.zeros(1), torch.ones(1)]])
num_samples = 128
num_iter = 2

# plot function f
import matplotlib.pyplot as plt
tx = torch.linspace(0,1,100)
plt.plot(tx.squeeze().numpy(),f(tx).squeeze().numpy())
plt.scatter(train_X.numpy(),train_Y.numpy(),c='r',s=100,marker='X')
plt.show()
# %%
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling.pathwise import draw_matheron_paths
NUM_PATHS = 1000
outcome_transform = Standardize(m=1)
gp = SingleTaskGP(train_X, train_Y, outcome_transform=outcome_transform) # gp model
mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # mll object
_ = fit_gpytorch_mll(mll) # fit mll hyperpara
best_f = train_Y.max()
paths = draw_matheron_paths(gp, torch.Size([NUM_PATHS]))
ves = VariationalEntropySearch(gp, best_f=best_f, optimal_inputs=train_X, optimal_outputs=train_Y, paths=paths)

# %%
betas = [0.01, 1.0, 5.0]
ks = [0.01, 1.0, 1.5]
plot_ves(ves, betas, ks, train_X, train_Y, f)
# %%
