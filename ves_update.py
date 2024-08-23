from __future__ import annotations
from copy import deepcopy
from typing import Any, Optional, Tuple, Callable, Union
import numpy as np
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling import SobolEngine
# from botorch.utils.sampling import optimize_posterior_samples
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
from torch import Tensor
from torch.special import digamma, polygamma

# NUM_OPTIMA = 100

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

def find_root_log_minus_digamma(
        intercept, 
        initial_guess, 
        tol=1e-5, 
        max_iter=int(1e3),
        lower_bound=torch.tensor(1e-8),
        upper_bound=torch.tensor(1e8)):
    """
    Find a root of the function log(x) - digamma(x) - intercept using a combination of
    the bisection method and Newton's method.

    Args:
    intercept (float or tensor): The constant value to subtract in the function.
    initial_guess (float or tensor): Initial guess for the root.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.

    Returns:
    float or tensor: Approximate root of the function.
    """
    def f(x):
        return torch.log(x) - digamma(x) - intercept

    # Step 1: Bisection method
    for _ in range(int(max_iter)):
        midpoint = (lower_bound + upper_bound) / 2.0
        f_mid = f(midpoint)

        if torch.abs(f_mid) < tol:
            return midpoint

        # Narrow down the interval
        if f(lower_bound) * f_mid < 0:
            upper_bound = midpoint
        else:
            lower_bound = midpoint

        # Switch to Newton's method when the interval is small enough
        if torch.abs(upper_bound - lower_bound) < 1e-2:
            x = midpoint
            break
    else:
        x = initial_guess  # If bisection did not converge, use the initial guess

    # Step 2: Newton's method
    for _ in range(int(max_iter)):
        value = f(x)
        derivative = 1/x - polygamma(1, x)  # derivative of the function

        # Newton's method update
        x_new = x - value / derivative

        # Check for convergence
        if torch.abs(x_new - x) < tol:
            return x_new

        x = x_new

    print("The root finding method does not converge. A default value 1.0 is assigned on k.")
    return torch.tensor(1.0)

class HalfVESGamma(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        paths,
        optimal_outputs: Tensor,
        k: Union[float, Tensor],
        beta: Union[float, Tensor]
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

    def forward(self, X: Tensor):
        """
        The forward function evaluates ESLB for fixed k and beta
        Follow Eq 3.8
        Args:
            X: batch_size x q=1 x dim input position
        Returns:
            output value
        """
        posterior_samples = self.paths(X.squeeze(1))
        improvement_term = torch.max(posterior_samples, self.best_f).unsqueeze(1)
        # This should be able to be logged, since it is per-sample
        max_value_term = (self.optimal_outputs - improvement_term).clamp_min(1e-3) 
        log_max_value = max_value_term.log()
        max_value_mean = max_value_term.nanmean(0)
        log_max_mean = log_max_value.nanmean(0)

        return ((self.k - 1) * log_max_mean + self.beta * max_value_mean).squeeze()

class VariationalEntropySearchGamma(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        paths,
        bounds: Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
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
        self.optimal_outputs = optimize_posterior_samples(
            paths, 
            bounds)
        self.paths = paths
        self.bounds = bounds

    def forward(self, X, num_iter: int = 64):
        """
        This VES class implements VES-Gamma, a special case of VES. 
        There are two steps: the first step is to design a new halfVES class, which
        is uniquely specified by the value of k and beta. A optimize_acqf optimizer 
        is implemented to find its optimal candidate. The second step is to use this
        selected optimal candidate to find the next k and beta, and generate a new 
        halfVES class. The iteration is repeated num_iter times.
        Args:
            X: The initial value for X batch_size x q x dim
            num_iter: number of iterations
        Returns:
            candidate: the optimal position for the solution
            acq_value: the value of its position
        """
        cur_X = X
        for i in range(num_iter):
            # Step 1: Find current optimal k and beta
            max_value_term = self.generate_max_value_term(cur_X)
            kval, betaval = self.find_k(max_value_term)
            halfVES = HalfVESGamma(self.model, 
                                   self.best_f, 
                                   self.paths,
                                   self.optimal_outputs,
                                   kval.item(), betaval.item())
            # Step 2: Given k and beta, find optimal X
            cur_X, acq_value = optimize_acqf(
                halfVES, 
                bounds=self.bounds.T,
                q=1,  # Number of candidates to optimize for
                num_restarts=5,
                raw_samples=20,  # Initial samples for optimization
            )
            if i % 10 == 0:
                print(f"Iteration {i}:")
                print(f"K: {kval}; beta {betaval}; X:{cur_X}, AF value: {acq_value}")
        return cur_X, acq_value
        
    def generate_max_value_term(self, X: Tensor):
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
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(1e-3) 
        # This should be able to be logged, since it is per-sample
        return max_value_term

    def find_k(self, max_value_term: Tensor):
        """
        This function evaluates the optimal values of k and beta
        Args:
            max_value_term: NUM_PATH x q x batch_size
        Return:
            k_vals: q x batch_size
            beta_vals: q x batch_size
        """
        A = max_value_term.nanmean(dim=0)
        B = (torch.log(max_value_term)).nanmean(dim=0)
        self.v = torch.log(A) - B
        k_vals = self.root_finding(self.v)
        beta_vals = k_vals / A
        return k_vals, beta_vals

    def root_finding(self, x: Tensor):
        """
        Root finding function to solve Eq 3.9; Non-differentiable(?)
        """
        res = np.zeros_like(x.flatten().detach().numpy())
        for i, xx in enumerate(x.flatten().detach().numpy()):
            res[i] = find_root_log_minus_digamma(torch.tensor(xx), initial_guess=torch.tensor(0.5))
        return torch.Tensor(res).reshape(x.shape)
    
if __name__ == "__main__":
    # Test VES on a trivial example (D=5)
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.sampling.pathwise import draw_matheron_paths
    from botorch.acquisition import ExpectedImprovement
    from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
    NUM_PATHS = 1024

    N = 4
    D = 5
    train_X = torch.rand(N, D)
    def f(X, noise=0.0):
        return -torch.norm(torch.sin(15*X)*X, dim=1)**2
    train_Y = f(train_X, noise=0.0).reshape(N, 1)
    bounds = torch.zeros(D, 2)
    bounds[:, 1] = 1
    outcome_transform = Standardize(m=1)
    gp = SingleTaskGP(train_X, train_Y, outcome_transform=outcome_transform) # gp model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # mll object
    _ = fit_gpytorch_mll(mll) # fit mll hyperpara
    best_f = train_Y.max()
    paths = draw_matheron_paths(gp, torch.Size([NUM_PATHS]))
    ves_model = VariationalEntropySearchGamma(gp, best_f=best_f, bounds=bounds, paths=paths)
    ei_model = ExpectedImprovement(gp, best_f)

    # Define an intial point for VES-Gamma
    X = torch.rand(1, 1, D)
    vescandidate, v = ves_model(X, num_iter=64)
    print(vescandidate, v)
    eicandidate, acq_value = optimize_acqf(
        ei_model, 
        bounds=bounds.T,
        q=1,  # Number of candidates to optimize for
        num_restarts=5,
        raw_samples=20,  # Initial samples for optimization
    )
    print(eicandidate, acq_value)
    print(f(vescandidate).item(), f(eicandidate).item())

