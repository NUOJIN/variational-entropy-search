from __future__ import annotations

import json
import math
import os
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Any, Optional, Tuple, Union, Callable
from zlib import adler32

import grpc
import numpy as np
import scipy
import torch
from bencherscaffold.bencher_pb2 import BenchmarkRequest
from bencherscaffold.bencher_pb2_grpc import BencherStub
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
# from tqdm import tqdm
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
# from botorch.utils.sampling import optimize_posterior_samples
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_gaussian_likelihood_with_gamma_prior
from botorch.optim import optimize_acqf
from botorch.sampling import SobolEngine
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.test_functions import Hartmann, Branin
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from scipy import optimize
from torch import Tensor


@contextmanager
def torch_random_seed(
        seed: int,
):
    """
    Sets the random seed for torch operations within the context.

    Parameters:
    seed (int): The random seed to be set.

    This function sets the random seed for torch operations within the context. After the context is exited, the random
    seed is reset to its original value.
    """
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)


class BenchmarkType(Enum):
    BOTORCH = 1
    BENCHER = 2
    GP_PRIOR_SAMPLE = 3


def get_objective(
        benchmark_name: str,
) -> Tuple[Callable[[Tensor], Tensor], int]:
    """

    Args:
        benchmark_name (str): the name of the benchmark

    Returns:
        Tuple[Callable[[Tensor], Tensor], int]: the objective function and the dimensionality of the problem

    """

    match benchmark_name:
        case "lasso-dna":
            benchmark_dim = 180
            benchmark_type = BenchmarkType.BENCHER
        case "mopta08":
            benchmark_dim = 124
            benchmark_type = BenchmarkType.BENCHER
        case "svm":
            benchmark_dim = 388
            benchmark_type = BenchmarkType.BENCHER
        case "mujoco-ant":
            benchmark_dim = 888
            benchmark_type = BenchmarkType.BENCHER
        case "mujoco-humanoid":
            benchmark_dim = 6392
            benchmark_type = BenchmarkType.BENCHER
        case "robotpushing":
            benchmark_dim = 14
            benchmark_type = BenchmarkType.BENCHER
        case "lasso-breastcancer":
            benchmark_dim = 10
            benchmark_type = BenchmarkType.BENCHER
        case "rover":
            benchmark_dim = 60
            benchmark_type = BenchmarkType.BENCHER
        case 'hartmann6':
            benchmark_dim = 6
            benchmark_type = BenchmarkType.BOTORCH
        case 'branin2':
            benchmark_dim = 2
            benchmark_type = BenchmarkType.BOTORCH
        case s if s.startswith('prior_sample_'):
            benchmark_dim = int(s.split('_')[2][:-1])
            sample_ls = float(s.split('_')[-1][2:])
            benchmark_type = BenchmarkType.GP_PRIOR_SAMPLE
        case _:
            raise ValueError("Invalid benchmark")

    def objective(
            x: Tensor,
    ) -> Tensor:
        """
        The objective function

        Args:
            x: the input

        Returns:
            Tensor: the output

        """
        if benchmark_type == BenchmarkType.BOTORCH:
            if args.benchmark == 'hartmann6':
                _f = Hartmann(negate=True)
                return _f(x)
            elif args.benchmark == 'branin2':

                branin_bounds = torch.tensor([[-5, 10], [0, 15]])
                x_eval = x * (branin_bounds[1] - branin_bounds[0]) + branin_bounds[0]

                _f = Branin(negate=True)
                return _f(x_eval)

        elif benchmark_type == BenchmarkType.BENCHER:
            stub = BencherStub(
                grpc.insecure_channel(f"localhost:50051")
            )
            assert x.ndim in [1, 2], 'x must be 1D or 2D'
            _x = x
            if x.ndim == 2:
                assert x.shape[0] == 1, 'x has to be essentially 1D'
                _x = x.squeeze(0)
            # add timeout to evaluation
            n_retries = 0
            failed = True
            while n_retries < 10:
                try:
                    res = stub.evaluate_point(
                        BenchmarkRequest(
                            benchmark=benchmark_name,
                            point={
                                'values': _x.tolist()
                            }
                        ),
                    )
                    failed = False
                    break
                except Exception as e:
                    print(f'error: {e}')
                    n_retries += 1
                    if n_retries == 10 and failed:
                        raise e
                    time.sleep(5)
            # negate the result since we are maximizing
            _res = -res.value
            # to torch.double
            return torch.tensor(_res, dtype=torch.double)
        elif benchmark_type == BenchmarkType.GP_PRIOR_SAMPLE:
            prior_sample_gp_covar_module = MaternKernel(
                nu=2.5,
                ard_num_dims=benchmark_dim,
            )
            prior_sample_gp_covar_module.lengthscale = torch.tensor(sample_ls)
            prior_sample_gp = SingleTaskGP(
                torch.empty(0, benchmark_dim, dtype=torch.double),
                torch.empty(0, 1, dtype=torch.double),
                covar_module=prior_sample_gp_covar_module,
            )

            with torch_random_seed(42):
                prior_sample_gp_path = draw_matheron_paths(
                    model=deepcopy(prior_sample_gp),
                    sample_shape=torch.Size([1]),
                )
            return prior_sample_gp_path(x.detach().reshape(1, -1)).detach().squeeze()
        else:
            raise ValueError("Invalid benchmark type")

    return objective, benchmark_dim


def get_gp(
        train_x: Tensor,
        train_y: Tensor,
        gp_lengthscale: Optional[float] = None,
        gp_noise: Optional[float] = None,
        gp_outputscale: Optional[float] = None
) -> SingleTaskGP:
    """
    Get a GP model with a Matern kernel and Gamma prior on the lengthscale.

    Args:
        train_x: the training x data
        train_y: the training y data
        gp_lengthscale: the lengthscale of the GP
        gp_noise: the noise of the GP
        gp_outputscale: the outputscale

    Returns:
        SingleTaskGP: the GP model

    """
    outcome_transform = Standardize(m=1)
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_prior=GammaPrior(1.5, 0.1)))
    if gp_lengthscale is not None:
        covar_module.base_kernel.lengthscale = gp_lengthscale
        covar_module.base_kernel.raw_lengthscale.requires_grad = False
    if gp_outputscale is not None:
        covar_module.outputscale = gp_outputscale
        covar_module.raw_outputscale.requires_grad = False
    likelihood = get_gaussian_likelihood_with_gamma_prior()
    if gp_noise is not None:
        likelihood.noise = gp_noise
    else:
        likelihood.noise = 1e-4
    likelihood.raw_noise.requires_grad = False

    gp = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=outcome_transform,
        covar_module=covar_module,
        likelihood=likelihood
    )
    return gp


def str2bool(
        v: str
):
    if v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise ValueError("Invalid boolean value")


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
        dimension=bounds.shape[0], scramble=True
    ).draw(raw_samples)
    # TODO add spray points
    # queries all samples on all candidates - output raw_samples * num_objectives * num_optima
    candidate_queries = paths.forward(candidate_set)
    num_optima = candidate_queries.shape[0]
    batch_size = candidate_queries.shape[1] if candidate_queries.ndim == 3 else 1
    argtop_candidates = candidate_queries.argsort(dim=-1, descending=True)[..., 0:num_restarts]

    # These are used as masks when retrieving the argmaxes
    X_argtop = candidate_set[argtop_candidates, :].requires_grad_(requires_grad=True)
    for i in range(maxiter):
        per_sample_outputs = paths.forward(X_argtop)
        grads = torch.autograd.grad(
            per_sample_outputs, X_argtop, grad_outputs=torch.ones_like(per_sample_outputs)
        )[0]
        X_argtop = torch.clamp(X_argtop + lr * grads, 0, 1)  # TODO fix bounds here

    per_sample_outputs = paths.forward(X_argtop).reshape(num_optima * batch_size, num_restarts)
    f_max = per_sample_outputs.max(axis=-1).values.reshape(num_optima, batch_size, 1)

    return f_max.detach()


def find_root_log_minus_digamma(
        intercept,
        initial_guess,
        tol=1e-5,
        lower_bound=1e-8,
        upper_bound=1e8
):
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

    def f(
            x
    ):
        return math.log(x) - scipy.special.digamma(x) - intercept

    def f_least_square(
            x
    ):
        return f(x) ** 2

    try:
        root_finding_result = optimize.minimize_scalar(
            f_least_square,
            bounds=(lower_bound, upper_bound),
            options={'xatol': tol}
        ).x
    except:
        root_finding_result = 1.0

    # root_finding_result = optimize.root_scalar(
    #    f,
    #    x0=initial_guess,
    #    rtol=tol
    # )
    return root_finding_result


class HalfVESGamma(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            paths,
            optimal_outputs: Tensor,
            k: Union[float, Tensor],
            beta: Union[float, Tensor],
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
            X: Tensor
    ):
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
        max_value_term = (self.optimal_outputs - improvement_term).clamp_min(self.clamp_min)
        log_max_value = max_value_term.log()
        improvement_mean = improvement_term.mean(0)
        log_max_mean = log_max_value.mean(0)
        return ((self.k - 1) * log_max_mean + self.beta * improvement_mean).squeeze()


class VariationalEntropySearchGamma(MCAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            paths,
            clamp_min: float,
            optimize_acqf_options: dict[str, Any] | None = None,
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
            bounds
        )
        self.paths = paths
        self.bounds = bounds
        self.clamp_min = clamp_min
        if optimize_acqf_options is None:
            optimize_acqf_options = {
                "num_restarts": 5,
                "raw_samples" : 1024,
                "options"     : {"sample_around_best": True}
            }
        self.optimize_acqf_options = optimize_acqf_options

    def forward(
            self,
            X,
            num_iter: int = 64,
            show_progress: bool = True
    ):
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
        assert num_iter > 0, "Number of iterations should be positive"

        cur_X = X
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
            new_X, acq_value = optimize_acqf(
                halfVES,
                bounds=self.bounds.T,
                q=1,  # Number of candidates to optimize for
                **self.optimize_acqf_options
            )
            if torch.norm(new_X - cur_X) < 1e-5:
                break
            cur_X = new_X
            self.paths = draw_matheron_paths(self.model, torch.Size([num_paths]))
            self.optimal_outputs = optimize_posterior_samples(
                self.paths,
                self.bounds
            )
            if show_progress and i % 5 == 0:
                print(f"Iteration {i}: K: {kval.item():.3e}; beta {betaval.item():.3e}; AF value: {acq_value:.3e}")
        return cur_X, acq_value, kval.item(), betaval.item()

    def generate_max_value_term(
            self,
            X: Tensor
    ):
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
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(self.clamp_min)
        # This should be able to be logged, since it is per-sample
        return max_value_term

    def find_k(
            self,
            max_value_term: Tensor
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
            x: Tensor
    ):
        """
        Root finding function to solve Eq 3.9; Non-differentiable(?)
        """
        res = np.zeros_like(x.flatten().detach().numpy())
        for i, intercept in enumerate(x.flatten().detach().numpy()):
            res[i] = find_root_log_minus_digamma(intercept, initial_guess=0.5)
        return torch.Tensor(res).reshape(x.shape)


class VariationalEntropySearchExponential(MCAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            paths,
            clamp_min: float,
            optimize_acqf_options: dict[str, Any] | None = None,
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
            bounds
        )
        self.paths = paths
        self.bounds = bounds
        self.clamp_min = clamp_min
        if optimize_acqf_options is None:
            optimize_acqf_options = {
                "num_restarts": 5,
                "raw_samples" : 1024,
                "options"     : {"sample_around_best": True}
            }
        self.optimize_acqf_options = optimize_acqf_options

    def forward(
            self,
            X,
            num_iter: int = 64
    ):
        """
        This VES class implements VES-Exp, a special case of VES, and expected to be
        equivalent to EI. 
        Args:
            X: The initial value for X batch_size x q x dim
            num_iter: number of iterations
        Returns:
            candidate: the optimal position for the solution
            acq_value: the value of its position
        """
        assert num_iter > 0, "Number of iterations should be positive"

        cur_X = X
        # solve beta (or lambda)
        betaval = torch.ones(cur_X.shape[0])
        kval = torch.ones_like(betaval)
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
        cur_X, acq_value = optimize_acqf(
            halfVES,
            bounds=self.bounds.T,
            q=1,  # Number of candidates to optimize for
            **self.optimize_acqf_options
        )

        return cur_X, acq_value, kval.item(), betaval.item()

    def generate_max_value_term(
            self,
            X: Tensor
    ):
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
        max_value_term = (self.optimal_outputs.squeeze(1) - improvement_term).clamp_min(self.clamp_min)
        # This should be able to be logged, since it is per-sample
        return max_value_term


if __name__ == "__main__":
    # Test VES on a trivial example (D=5)

    argparse = ArgumentParser()
    argparse.add_argument("--num_paths", type=int, default=64, help="Number of paths to sample")
    argparse.add_argument("--num_iter", type=int, default=50, help="Number of iterations for VES")
    argparse.add_argument("--num_bo_iter", type=int, default=500)
    argparse.add_argument("--n_init", type=int, default=20)
    argparse.add_argument("--clamp_min", type=float, default=1e-10)
    argparse.add_argument("--acqf_raw_samples", type=int, default=512)
    argparse.add_argument("--acqf_num_restarts", type=int, default=5)
    argparse.add_argument("--sample_around_best", type=str2bool, default=True)
    argparse.add_argument("--run_ei", type=str2bool, default=False)
    argparse.add_argument("--run_mes", type=str2bool, default=False)
    argparse.add_argument("--exponential_family", type=str2bool, default=False)
    argparse.add_argument("--set_lengthscale", type=float, default=None)
    argparse.add_argument("--set_noise", type=float, default=None)
    argparse.add_argument("--set_outputscale", type=float, default=None)

    argparse.add_argument(
        "--benchmark", type=str, choices=[
            "lasso-dna",
            "mopta08",
            "svm",
            "mujoco-ant",
            "mujoco-humanoid",
            "robotpushing",
            "lasso-breastcancer",
            "rover",
            "hartmann6",
            "branin2",
            "prior_sample_10d_ls0.5",
            "prior_sample_10d_ls1",
            "prior_sample_10d_ls2",
            "prior_sample_50d_ls0.5",
            "prior_sample_50d_ls1",
            "prior_sample_50d_ls2",
            "prior_sample_2d_ls0.5",
            "prior_sample_2d_ls1",
            "prior_sample_2d_ls2",
        ],
        required=True
    )

    args = argparse.parse_args()

    # When both run_ei and run_mes are True, run_ei will be executed
    if args.run_ei and args.run_mes:
        args.run_mes = False

    # Define the objective function
    objective, D = get_objective(benchmark_name=args.benchmark)

    args_dir = vars(args)
    # calculate run_dir hash with adler32
    run_dir = f"{adler32(json.dumps(args_dir).encode())}"
    timestamp_ms = int(time.time_ns() / 1e6)
    run_dir = f"{timestamp_ms}_{run_dir}"
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        run_dir = f'{run_dir}_{os.environ["SLURM_ARRAY_TASK_ID"]}'

    os.makedirs(f"runs/{run_dir}", exist_ok=True)

    # save args.json to run_dir
    with open(f"runs/{run_dir}/args.json", "w") as file:
        json.dump(args_dir, file)

    num_paths = args.num_paths
    benchmark_name = args.benchmark
    clamp_min = args.clamp_min
    run_ei = args.run_ei
    run_mes = args.run_mes
    gp_lengthscale = args.set_lengthscale
    gp_noise = args.set_noise
    gp_outputscale = args.set_outputscale

    if args.exponential_family:
        ves_class = VariationalEntropySearchExponential
    else:
        ves_class = VariationalEntropySearchGamma
    acqf_options = {
        "num_restarts": args.acqf_num_restarts,
        "raw_samples" : args.acqf_raw_samples,
        "options"     : {"sample_around_best": args.sample_around_best}
    }

    n_init = args.n_init
    train_x_ves = torch.rand(n_init, D, dtype=torch.double)
    train_y_ves = torch.Tensor([objective(x) for x in train_x_ves]).unsqueeze(1).to(torch.double)

    if run_ei:
        train_x_ei = train_x_ves.clone()
        train_y_ei = train_y_ves.clone()
    if run_mes:
        train_x_mes = train_x_ves.clone()
        train_y_mes = train_y_ves.clone()

    bounds = torch.zeros(D, 2)
    bounds[:, 1] = 1

    # partial function to get the GP that already has the lengthscale, noise, and outputscale set
    _get_gp = partial(
        get_gp,
        gp_lengthscale=gp_lengthscale,
        gp_noise=gp_noise,
        gp_outputscale=gp_outputscale
    )

    if run_ei:
        gp_ei = _get_gp(train_x_ei, train_y_ei)
        mll_ei = ExactMarginalLogLikelihood(gp_ei.likelihood, gp_ei)  # mll object
        fit_gpytorch_mll(mll_ei)  # fit mll hyperparameters
        ei_model = LogExpectedImprovement(gp_ei, train_y_ei.max())
    elif run_mes:
        gp_mes = _get_gp(train_x_mes, train_y_mes)
        mll_mes = ExactMarginalLogLikelihood(gp_mes.likelihood, gp_mes) # mll object
        fit_gpytorch_mll(mll_mes) # fit mll hyperparameters
        candidate_set = SobolEngine(dimension=bounds.shape[0], scramble=True).draw(2048)
        mes_model = qMaxValueEntropy(gp_mes, candidate_set)
    else:
        gp_ves = _get_gp(train_x_ves, train_y_ves)
        mll_ves = ExactMarginalLogLikelihood(gp_ves.likelihood, gp_ves)  # mll object
        fit_gpytorch_mll(mll_ves)  # fit mll hyperparameters
        paths = draw_matheron_paths(gp_ves, torch.Size([num_paths]))
        ves_model = ves_class(
            gp_ves,
            best_f=train_y_ves.max(),
            bounds=bounds,
            paths=paths,
            clamp_min=clamp_min,
            acqf_options=acqf_options,
        )
        k_vals = []
        beta_vals = []

    for i in range(args.num_bo_iter):
        print(f"+++ Iteration {i} +++")
        # Define an intial point for VES-Gamma
        X = torch.rand(1, 1, D)
        if run_ei:
            ei_candidate, acq_value = optimize_acqf(
                ei_model,
                bounds=bounds.T,
                q=1,  # Number of candidates to optimize for
                num_restarts=args.acqf_num_restarts,
                raw_samples=args.acqf_raw_samples,
            )
            train_x_ei = torch.cat([train_x_ei, ei_candidate], dim=0)
            f_ei = objective(ei_candidate)
            print(
                f"EI: cand={ei_candidate}, acq_val={acq_value:.3e}, f_val={f_ei.item():.3e}, f_max={train_y_ei.max()}"
            )
            train_y_ei = torch.cat([train_y_ei, f_ei.reshape(1, 1)], dim=0)
            # save the results
            np.save(f"runs/{run_dir}/train_x_ei.npy", train_x_ei.detach().numpy())
            np.save(f"runs/{run_dir}/train_y_ei.npy", train_y_ei.detach().numpy())

            gp_ei = _get_gp(train_x_ei, train_y_ei)
            mll_ei = ExactMarginalLogLikelihood(gp_ei.likelihood, gp_ei)  # mll object
            fit_gpytorch_mll(mll_ei)  # fit mll hyperpara
            ei_model = LogExpectedImprovement(gp_ei, train_y_ei.max())
        elif run_mes:
            mes_candidate, acq_value = optimize_acqf(
                mes_model,
                bounds=bounds.T,
                q=1,  # Number of candidates to optimize for
                num_restarts=args.acqf_num_restarts,
                raw_samples=args.acqf_raw_samples,
            )
            train_x_mes = torch.cat([train_x_mes, mes_candidate], dim=0)
            f_mes = objective(mes_candidate)
            print(
                f"MES: cand={mes_candidate}, acq_val={acq_value:.3e}, f_val={f_mes.item():.3e}, f_max={train_y_mes.max()}"
            )
            train_y_mes = torch.cat([train_y_mes, f_mes.reshape(1, 1)], dim=0)
            # save the results
            np.save(f"runs/{run_dir}/train_x_mes.npy", train_x_mes.detach().numpy())
            np.save(f"runs/{run_dir}/train_y_mes.npy", train_y_mes.detach().numpy())

            gp_mes = _get_gp(train_x_mes, train_y_mes)
            mll_mes = ExactMarginalLogLikelihood(gp_mes.likelihood, gp_mes)
            fit_gpytorch_mll(mll_mes)
            mes_model = qMaxValueEntropy(gp_mes, candidate_set)
        else:
            ves_candidate, v, k_val, beta_val = ves_model(X, num_iter=args.num_iter)
            k_vals.append(k_val)
            beta_vals.append(beta_val)
            train_x_ves = torch.cat([train_x_ves, ves_candidate], dim=0)
            f_ves = objective(ves_candidate)
            print(f"VES: cand={ves_candidate}, acq_val={v:.3e}, f_val={f_ves.item():.3e}, f_max={train_y_ves.max()}")
            train_y_ves = torch.cat([train_y_ves, f_ves.reshape(1, 1)], dim=0)
            # save the results
            np.save(f"runs/{run_dir}/train_x_ves.npy", train_x_ves.detach().numpy())
            np.save(f"runs/{run_dir}/train_y_ves.npy", train_y_ves.detach().numpy())

            # save k_vals and beta_vals
            np.save(f"runs/{run_dir}/k_vals.npy", np.array(k_vals))
            np.save(f"runs/{run_dir}/beta_vals.npy", np.array(beta_vals))

            gp_ves = _get_gp(train_x_ves, train_y_ves)

            mll_ves = ExactMarginalLogLikelihood(gp_ves.likelihood, gp_ves)  # mll object
            fit_gpytorch_mll(mll_ves)  # fit mll hyperpara

            paths = draw_matheron_paths(gp_ves, torch.Size([num_paths]))
            ves_model = ves_class(
                gp_ves,
                best_f=train_y_ves.max(),
                bounds=bounds,
                paths=paths,
                clamp_min=clamp_min,
                acqf_options=acqf_options,
            )
