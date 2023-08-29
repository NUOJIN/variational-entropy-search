from __future__ import annotations
from scipy.optimize import minimize, LinearConstraint
import numpy as np
from torch import nn
from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from botorch.acquisition import ExpectedImprovement,UpperConfidenceBound,qKnowledgeGradient
import torch
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.pathwise import draw_matheron_paths, SamplePath
from botorch.sampling import SobolEngine

import tqdm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf

class VariationalEntropySearch(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        X_init: Tensor,
        bounds: Tensor,
        beta_lst: Tensor,
        D_t: Tensor,
        num_EM_iter: Optional[int] = 16
    ) -> None:
        super().__init__(model=model)
        self.model = model
        self.bounds = bounds
        self.D_t = D_t # D_t is a tensor of y values
        self.beta_lst = beta_lst
        #self.eta = torch.from_numpy(np.eye(len(beta_lst) + 1)[2]).type(torch.double)
        self.eta = torch.from_numpy(np.ones(len(beta_lst)+1)/4).type(torch.double)
        self.x = nn.Parameter(X_init, requires_grad=True)
        
        self.eta_lst = []

    def forward(self,EM_iter_num=64):
        for _ in range(EM_iter_num):
            old_x = self.x
            self.expectation_step()
            self.maximization_step()
            if torch.norm(self.x.detach()-old_x.detach()) < 1e-8:
                break
            print('Finish EM iteration ',_)

    def sample(self, N=100): # it computes N samples from current posterior and returns two vector tensors: y_star and y_x based on given x
        r"""Return a tuple of tensors containing joint samples y_star and y_x.

            Args:
                N: number of path samples

            Returns:
                Tensor: A `N` tensor y_star and a `N` tensor y_x
            """

        paths = draw_matheron_paths(self.model, sample_shape=torch.Size([N]))
        y_star = optimize_posterior_samples(paths, bounds=self.bounds) 
        y_x = paths.forward(self.x.unsqueeze(0))

        return y_star.squeeze(), y_x.squeeze()

    def h_func(self, y_x) -> Tensor: 
        r"""Return a tensor of h function values given y_x.

            Args:
                y_x: A `N` tensor representing y_x from N samples given x

            Returns:
                Tensor: A `N x K` tensor with (n, k) entry representing h_k value of n-th sample
            """
        y_best = torch.max(self.D_t)
        ei = torch.maximum(y_best, y_x)

        K = len(self.beta_lst) + 1
        N = len(y_x)
        h = torch.zeros(size=(N, K-1))

        for n in range(N):
          for k in range(K-1):
            new_Dt = torch.cat((self.D_t, Tensor([y_x[n]]).to(torch.double)))
            mean = torch.mean(new_Dt)
            std = torch.std(new_Dt)
            h[n, k] = mean + self.beta_lst[k] * std

        return torch.cat((ei.unsqueeze(1), h), 1)

    def eslb(self, 
             eta: Tensor, # A `K` tensor representing the value of eta
             y_star: Tensor, # A `N` tensor representing y^* from N samples
             y_x: Tensor, # A `N` tensor representing y_x from N samples given x
             ):# evaluate ESLB value given x, eta
        r"""Evaluate ESLB given eta and fixed N y^* and y_x values.

            Args:
                eta: A `K` tensor representing the value of eta
                y_star: A `N` tensor representing y^* from N samples
                y_x: A `N` tensor representing y_x from N samples given x

            Returns:
                Tensor: A scalar result representing ESLB value
            """
        K = len(self.beta_lst) + 1
        y_star = torch.kron(y_star, torch.ones(size=(K,1))).T # make y_star a `N x K` matrix
        h = self.h_func(y_x) # h function returning h function values given different y_x, output: one Tensor with size 'N x K', K is the number of h_k
        ind = y_star > h # return an indicator matrix indicating if y^* larger than h_k given y_x, return a Tensor with size `N x K`
        ind[:, 0] = True # ei part is always true
        
        return torch.mean(torch.log((torch.exp(- y_star + h) * ind) @ eta))
            
    def expectation_step(self, N=500): # update eta to maximize eslb
        y_star, y_x = self.sample(N) # sample function for sampling joint y^* and y_x, output: two Tensor with length N
        y_x = y_x.detach()
        K = len(self.beta_lst) + 1
        def f(eta):
          return -self.eslb(eta,y_star,y_x).item()
        ## construct linear constraint for eta
        #cnst_M = np.vstack((np.ones(K),np.eye(K)))
        cnst_eq_A = np.expand_dims(np.ones(K), axis=0)
        cnst_ineq_A = np.eye(K)
        
        cnst_eq = LinearConstraint(cnst_eq_A, np.array([1]), np.array([1]))
        cnst_ineq = LinearConstraint(cnst_ineq_A, np.zeros(K), np.ones(K))
        #cnst = LinearConstraint(cnst_M, np.eye(K+1)[0], np.ones(K+1))
        res = minimize(f, self.eta.detach().numpy(), constraints=[cnst_eq, cnst_ineq])
        self.eta = torch.from_numpy(res.x).type(torch.double)
        self.eta_lst.append(res.x)
            
    def maximization_step(self, batch_size=16, epochs=50, return_losses=False): # optimizing x should apply re-parameterization trick
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas= (0.9, 0.99))
        
        losses = []
       
        
# =============================================================================
#         for _ in range(iter_num):
#           opt.zero_grad()
#           y_star, y_x = self.sample()
#           loss = -self.eslb(self.eta, y_star, y_x)
#           loss.backward()
#           opt.step()
#           losses.append(np.copy(loss.detach().numpy()))
#           with torch.no_grad():
#             for i in range(len(self.x)):
#                 self.x[i].clamp_(self.bounds[0, i], self.bounds[1, i])
#         if return_losses:
#           return losses
#         return
# =============================================================================
    
        for _ in range(epochs):
            paths = draw_matheron_paths(self.model, sample_shape=torch.Size([batch_size]))
            y_star = optimize_posterior_samples(paths, bounds=self.bounds).squeeze()
            y_x = paths.forward(self.x.unsqueeze(0)).squeeze()
            loss = -self.eslb(self.eta, y_star, y_x)
    
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(np.copy(loss.detach().numpy()))
        if return_losses:
          return losses
        return
    
def optimize_posterior_samples(
    paths: SamplePath,
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
    X_argtop = candidate_set[argtop_candidates, :].requires_grad_(requires_grad=True)
    for i in range(maxiter):
        per_sample_outputs = paths.forward(X_argtop)
        grads = torch.autograd.grad(
            per_sample_outputs, X_argtop, grad_outputs=torch.ones_like(per_sample_outputs))[0]
        X_argtop = torch.clamp(X_argtop + lr * grads, 0, 1)  # TODO fix bounds here
    
    per_sample_outputs = paths.forward(X_argtop).reshape(num_optima * batch_size, num_restarts)
    f_max = per_sample_outputs.max(axis=-1).values.reshape(num_optima, batch_size, 1)
    
    return f_max.detach()

if __name__ == '__main__':
    
    train_X = 2*torch.rand(5, 2, dtype=torch.double)-1.0
    def f(X,noise=0.0):
        return 1 - (X - 0.5).norm(dim=-1, keepdim=True)
    train_Y = f(train_X,noise=0.1)
    bounds = torch.stack([torch.zeros(2)-2, 2*torch.ones(2)]).to(torch.double)
    num_samples = 128
    num_iter = 2
    
    # plot function f
    import matplotlib.pyplot as plt
    nx,ny = 101,101
    X1,X2 = torch.linspace(bounds[0,0],bounds[1,0],nx),torch.linspace(bounds[0,1],bounds[1,1],ny)
    Xx,Xy = torch.meshgrid(X1,X2)
    f_input = torch.vstack((Xx.flatten(),Xy.flatten())).T
    Yz = f(f_input).reshape(nx,ny)
    
    plt.contourf(Xx,Xy,Yz,25)
    plt.colorbar()
    plt.plot(train_X[:,0],train_X[:,1],'kx',mew=3)
    plt.show()
    
    gp = SingleTaskGP(train_X, train_Y) # gp model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # mll object
#    fit_gpytorch_mll(mll) # fit mll hyperpara
    
    ei = ExpectedImprovement(gp,best_f=train_Y.max())
    kg = qKnowledgeGradient(gp,num_fantasies=128)
    ucb = UpperConfidenceBound(gp,beta=1.0)
    ves = VariationalEntropySearch(
        model = gp, 
        X_init = Tensor([0.0, 0.0]).type(torch.double), 
        bounds = Tensor([[-2, -2], [2, 2]]).type(torch.double), 
        beta_lst = Tensor([0.01, 0.1, 0.5]).type(torch.double), 
        D_t = train_Y.squeeze())
    
    ## optimize acqf
    # ei
#     ei_candidate, _ = optimize_acqf(
#       ei, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
#     )
#     ei_x,ei_y = ei_candidate,f(ei_candidate)
#     print(ei_x,ei_y.item())
    
    # # kg
    # kg_candidate, _ =optimize_acqf(
    #   kg, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    # )
    # kg_x,kg_y = kg_candidate,f(kg_candidate)
    # print(kg_x,kg_y.item())
    
    # # ucb
    # ucb_candidate, _ =optimize_acqf(
    #   ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    # )
    # ucb_x,ucb_y = ucb_candidate,f(ucb_candidate)
    # print(ucb_x,ucb_y.item())
    
    # ves
    ves(EM_iter_num=5)
    ves_x,ves_y = ves.x,f(ves.x)
    print(ves_x,ves_y.item())
    