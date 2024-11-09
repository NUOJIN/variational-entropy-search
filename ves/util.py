import math
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from typing import Optional, Any, Tuple, Callable

import gpytorch
import grpc
import scipy
import torch
from bencherscaffold.bencher_pb2 import BenchmarkRequest
from bencherscaffold.bencher_pb2_grpc import BencherStub
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.utils.gpytorch_modules import get_gaussian_likelihood_with_gamma_prior
from botorch.sampling.pathwise import MatheronPath, draw_matheron_paths
from botorch.test_functions import Hartmann, Levy, Griewank, Branin
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.models import GP
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator.utils.errors import NotPSDError
from scipy.optimize import optimize
from torch import Tensor, Size
from torch.quasirandom import SobolEngine

AVAILABLE_BENCHMARKS = ["lasso-dna",
                        "lasso-high",
                        "lasso-hard",
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
                        "prior_sample_100d_ls0.5",
                        "prior_sample_100d_ls1",
                        "prior_sample_100d_ls2",
                        "prior_sample_2d_ls0.5",
                        "prior_sample_2d_ls1",
                        "prior_sample_2d_ls2",
                        "prior_sample_2d_ls0.1",
                        "prior_sample_2d_ls0.05",
                        "mujoco-halfcheetah",
                        "mujoco-walker",
                        "schwefel100",
                        "schwefel300",
                        "schwefel500",
                        "levy100",
                        "levy300",
                        "levy500",
                        "griewank100",
                        "griewank300",
                        "griewank500", ]


def str2bool(
        string_value: str
) -> bool:
    """
    Convert string to boolean

    Args:
        string_value: string to convert

    Returns:
        boolean value

    """
    if string_value.lower() in ['true', '1']:
        return True
    elif string_value.lower() in ['false', '0']:
        return False
    else:
        raise ValueError("Invalid boolean value")


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


def fit_mll_with_adam_backup(
        mll: ExactMarginalLogLikelihood,
) -> None:
    """
    Fit the likelihood using BoTorch's fit_mll but use Adam if the original optimization fails.

    Args:
        mll: The marginal log likelihood object.

    Returns:
        None

    """
    with gpytorch.settings.cholesky_max_tries(9):
        try:
            fit_gpytorch_mll(mll)
        except NotPSDError as e:
            try:
                warnings.warn(f"Error fitting MLL with L-BFGS: {e}. Running Adam-based optimization...")
                optimizer = torch.optim.Adam(mll.parameters(), lr=0.1)
                mll.train()
                model = mll.model
                for i in range(100):
                    optimizer.zero_grad()
                    output = mll.model(*model.train_inputs)
                    loss = -mll(output, model.train_targets)
                    loss.backward()
                    optimizer.step()
                mll.eval()
            except NotPSDError:
                warnings.warn("Adam optimizer failed to converge. Skipping model fitting.")
                mll.eval()


def get_gp(
        train_x: Tensor,
        train_y: Tensor,
        gp_lengthscale: Optional[float] = None,
        gp_noise: Optional[float] = None,
        gp_outputscale: Optional[float] = None,
        lengthscale_prior: Optional[str] = None,
) -> SingleTaskGP:
    """
    Get a GP model with a Matern kernel and Gamma prior on the lengthscale.

    Args:
        train_x: the training x data
        train_y: the training y data
        gp_lengthscale: the lengthscale of the GP
        gp_noise: the noise of the GP
        gp_outputscale: the outputscale
        lengthscale_prior: the prior on the lengthscale, choices are "bounce" and "vbo"

    Returns:
        SingleTaskGP: the GP model

    """
    assert lengthscale_prior in [None, "bounce", "vbo"], "Invalid lengthscale prior"
    outcome_transform = Standardize(m=1)

    D = train_x.shape[-1]

    if lengthscale_prior == "bounce" or lengthscale_prior is None:
        _lengthscale_prior = GammaPrior(3.0, 6.0)
    elif lengthscale_prior == "vbo":
        _lengthscale_prior = LogNormalPrior(math.sqrt(2) + math.log(D) / 2, math.sqrt(3))

    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_prior=_lengthscale_prior))
    if gp_lengthscale is not None:
        covar_module.base_kernel.lengthscale = gp_lengthscale
        covar_module.base_kernel.raw_lengthscale.requires_grad = False
    if gp_outputscale is not None:
        covar_module.outputscale = gp_outputscale
        covar_module.raw_outputscale.requires_grad = False
    likelihood = get_gaussian_likelihood_with_gamma_prior()
    if gp_noise is not None:
        # TODO weird hack, we need to set gp_noise to allow for noise optimization
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


def robust_draw_matheron_paths(
        model: GP,
        sample_shape: Size,
        **kwargs: Any,
) -> MatheronPath:
    """
    Wrapper around draw_matheron_paths that sets the number of Cholesky decomposition tries to 9.

    Args:
        *args: the arguments to draw_matheron_paths
        **kwargs: the keyword arguments to draw_matheron_paths

    Returns:
        MatheronPath: the Matheron path

    """
    with gpytorch.settings.cholesky_max_tries(9):
        return draw_matheron_paths(model, sample_shape, **kwargs)


def find_root_log_minus_digamma(
        intercept,
        tol=1e-5,
        lower_bound=1e-8,
        upper_bound=1e8
):
    """
    Find a root of the function log(x) - digamma(x) - intercept using a combination of
    the bisection method and Newton's method.

    Args:
    intercept (float or tensor): The constant value to subtract in the function.
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

    return root_finding_result


def optimize_posterior_samples(
        paths,
        bounds: Tensor,
        raw_samples: Optional[int] = 2048,
        num_restarts: Optional[int] = 10,
        maxiter: int = 100,
        lr: float = 2.5e-4,
        device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    r"""Cheaply optimizes posterior samples by random querying followed by vanilla
    gradient descent on the best num_restarts points.

    Args:
        paths: Tample paths from the GP which can be called via forward()
        x: evaluation position for y_x
        bounds: The bounds on the search space.
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
    ).draw(raw_samples).to(device=device)
    # TODO add spray points
    # queries all samples on all candidates - output raw_samples * num_objectives * num_optima
    candidate_queries = paths.forward(candidate_set)
    num_optima = candidate_queries.shape[0]
    batch_size = candidate_queries.shape[1] if candidate_queries.ndim == 3 else 1
    argtop_candidates = candidate_queries.topk(dim=-1, k=num_restarts)[1]

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


class BenchmarkType(Enum):
    BOTORCH = 1
    BENCHER = 2
    GP_PRIOR_SAMPLE = 3


def get_objective(
        benchmark_name: str,
        device: torch.device,
) -> Tuple[Callable[[Tensor], Tensor], int]:
    """

    Args:
        benchmark_name (str): the name of the benchmark
        device (torch.device): the device

    Returns:
        Tuple[Callable[[Tensor], Tensor], int]: the objective function and the dimensionality of the problem

    """

    match benchmark_name:
        case "lasso-dna":
            benchmark_dim = 180
            benchmark_type = BenchmarkType.BENCHER
        case "lasso-high":
            benchmark_dim = 300
            benchmark_type = BenchmarkType.BENCHER
        case "lasso-hard":
            benchmark_dim = 1000
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
        case "mujoco-halfcheetah":
            benchmark_dim = 102
            benchmark_type = BenchmarkType.BENCHER
        case "mujoco-walker":
            benchmark_dim = 102
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
        case 'levy100':
            benchmark_dim = 100
            benchmark_type = BenchmarkType.BOTORCH
        case 'levy300':
            benchmark_dim = 300
            benchmark_type = BenchmarkType.BOTORCH
        case 'levy500':
            benchmark_dim = 500
            benchmark_type = BenchmarkType.BOTORCH
        case 'griewank100':
            benchmark_dim = 100
            benchmark_type = BenchmarkType.BOTORCH
        case 'griewank300':
            benchmark_dim = 300
            benchmark_type = BenchmarkType.BOTORCH
        case 'griewank500':
            benchmark_dim = 500
            benchmark_type = BenchmarkType.BOTORCH
        case 'schwefel100':
            benchmark_dim = 100
            benchmark_type = BenchmarkType.BOTORCH
        case 'schwefel300':
            benchmark_dim = 300
            benchmark_type = BenchmarkType.BOTORCH
        case 'schwefel500':
            benchmark_dim = 500
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
            if benchmark_name == 'hartmann6':
                _f = Hartmann(negate=True)
                return _f(x)
            elif benchmark_name.startswith('levy'):
                # name is something like levy300
                dim = int(benchmark_name[4:])
                levy_bounds = torch.tensor([[-10, 10]] * dim, dtype=torch.double, device=device).T
                x_eval = x * (levy_bounds[1] - levy_bounds[0]) + levy_bounds[0]

                _f = Levy(negate=True, dim=dim)
                return _f(x_eval)
            elif benchmark_name.startswith('griewank'):
                # name is something like griewank300
                dim = int(benchmark_name[8:])
                griewank_bounds = torch.tensor([[-600, 600]] * dim, dtype=torch.double, device=device).T
                x_eval = x * (griewank_bounds[1] - griewank_bounds[0]) + griewank_bounds[0]

                _f = Griewank(negate=True, dim=dim)
                return _f(x_eval)
            elif benchmark_name.startswith('schwefel'):
                # name is something like schwefel300
                dim = int(benchmark_name[8:])
                schwefel_bounds = torch.tensor([[-500, 500]] * dim, dtype=torch.double, device=device).T
                x_eval = x * (schwefel_bounds[1] - schwefel_bounds[0]) + schwefel_bounds[0]

                def schwefel(
                        x: Tensor,
                        dim: int,
                        negate: bool
                ) -> Tensor:
                    res = 418.9829 * dim - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))
                    return -res if negate else res

                return schwefel(x_eval, dim, True)

            elif benchmark_name == 'branin2':

                branin_bounds = torch.tensor([[-5, 10], [0, 15]], dtype=torch.double, device=device).T
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
            return torch.tensor(_res, dtype=torch.double, device=device)
        elif benchmark_type == BenchmarkType.GP_PRIOR_SAMPLE:
            prior_sample_gp_covar_module = MaternKernel(
                nu=2.5,
                ard_num_dims=benchmark_dim,
            )
            prior_sample_gp_covar_module.lengthscale = torch.tensor(sample_ls)
            prior_sample_gp = SingleTaskGP(
                torch.empty(0, benchmark_dim, dtype=torch.double, device=device),
                torch.empty(0, 1, dtype=torch.double, device=device),
                covar_module=prior_sample_gp_covar_module,
            )

            with torch_random_seed(42):
                prior_sample_gp_path = robust_draw_matheron_paths(
                    model=deepcopy(prior_sample_gp),
                    sample_shape=torch.Size([1]),
                )
            return prior_sample_gp_path(x.detach().reshape(1, -1)).detach().squeeze()
        else:
            raise ValueError("Invalid benchmark type")

    return objective, benchmark_dim
