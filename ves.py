from __future__ import annotations

import io
import json
import os
import tarfile
import time
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from typing import Any, Union
from zlib import adler32

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling import SobolEngine
from botorch.sampling.pathwise import MatheronPath
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from util import (
    str2bool,
    get_gp,
    fit_mll_with_adam_backup,
    robust_draw_matheron_paths,
    optimize_posterior_samples, find_root_log_minus_digamma, get_objective,
)


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
            paths: MatheronPath,
            clamp_min: float,
            optimize_acqf_options: dict[str, Any] | None = None,
            bounds: Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
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
            if torch.norm(new_X - cur_X) < self.stop_tolerance_coeff * self.bounds.size(0):
                break
            cur_X = new_X
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
        dtype, device = x.dtype, x.device
        x_np = x.flatten().detach().cpu().numpy()
        res = np.zeros_like(x_np)
        for i, intercept in enumerate(x_np):
            res[i] = find_root_log_minus_digamma(intercept)
        return torch.Tensor(res).reshape(x.shape).to(dtype=dtype, device=device)


class VariationalEntropySearchExponential(MCAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            paths,
            clamp_min: float,
            optimize_acqf_options: dict[str, Any] | None = None,
            bounds: Tensor = torch.Tensor([[torch.zeros(1), torch.ones(1)]]),
            device: torch.device = torch.device("cpu"),
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
            bounds,
            device=device
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
    argparse.add_argument("--lengthscale_prior", choices=["bounce", "vbo"], default="bounce")
    argparse.add_argument("--stop_tolerance_coeff", type=float, default=1e-5)
    argparse.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    argparse.add_argument(
        "--benchmark", type=str, choices=[
            "lasso-dna",
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
            "griewank500",
        ],
        required=True
    )

    args = argparse.parse_args()

    num_paths = args.num_paths
    benchmark_name = args.benchmark
    clamp_min = args.clamp_min
    run_ei = args.run_ei
    run_mes = args.run_mes
    gp_lengthscale = args.set_lengthscale
    gp_noise = args.set_noise
    gp_outputscale = args.set_outputscale
    lengthscale_prior = args.lengthscale_prior
    stop_tolerance_coeff = args.stop_tolerance_coeff
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # When both run_ei and run_mes are True, run_ei will be executed
    if args.run_ei and args.run_mes:
        args.run_mes = False

    # Define the objective function
    objective, D = get_objective(benchmark_name=args.benchmark, device=device)

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
    train_x_ves = torch.rand(n_init, D, dtype=torch.double, device=device)
    train_y_ves = torch.Tensor([objective(x) for x in train_x_ves]).unsqueeze(1).to(dtype=torch.double, device=device)

    if run_ei:
        train_x_ei = train_x_ves.clone()
        train_y_ei = train_y_ves.clone()
    if run_mes:
        train_x_mes = train_x_ves.clone()
        train_y_mes = train_y_ves.clone()

    bounds = torch.zeros(D, 2, dtype=torch.double, device=device)
    bounds[:, 1] = 1

    # partial function to get the GP that already has the lengthscale, noise, and outputscale set
    _get_gp = partial(
        get_gp,
        gp_lengthscale=gp_lengthscale,
        gp_noise=gp_noise,
        gp_outputscale=gp_outputscale,
        lengthscale_prior=lengthscale_prior,
    )

    if run_ei:
        gp_ei = _get_gp(train_x_ei, train_y_ei)
        mll_ei = ExactMarginalLogLikelihood(gp_ei.likelihood, gp_ei)  # mll object
        fit_mll_with_adam_backup(mll_ei)  # fit mll hyperparameters
        ei_model = LogExpectedImprovement(gp_ei, train_y_ei.max())
    elif run_mes:
        gp_mes = _get_gp(train_x_mes, train_y_mes)
        mll_mes = ExactMarginalLogLikelihood(gp_mes.likelihood, gp_mes)  # mll object
        fit_mll_with_adam_backup(mll_mes)  # fit mll hyperparameters
        candidate_set = SobolEngine(dimension=bounds.shape[0], scramble=True).draw(2048).to(device=device)
        mes_model = qMaxValueEntropy(gp_mes, candidate_set)
    else:
        gp_ves = _get_gp(train_x_ves, train_y_ves)
        mll_ves = ExactMarginalLogLikelihood(gp_ves.likelihood, gp_ves)  # mll object
        fit_mll_with_adam_backup(mll_ves)  # fit mll hyperparameters
        paths = robust_draw_matheron_paths(gp_ves, torch.Size([num_paths]))
        ves_model = ves_class(
            gp_ves,
            best_f=train_y_ves.max(),
            bounds=bounds,
            paths=paths,
            clamp_min=clamp_min,
            acqf_options=acqf_options,
            stop_tolerance_coeff=stop_tolerance_coeff,
            device=device,
        )
        k_vals = []
        beta_vals = []

    start_time = time.time()

    for bo_iter in range(args.num_bo_iter):
        print(f"+++ Iteration {bo_iter} +++")
        # Define an intial point for VES-Gamma
        X = torch.rand(1, 1, D, dtype=torch.double, device=device)
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
            np.save(f"runs/{run_dir}/train_x_ei.npy", train_x_ei.detach().cpu().numpy())
            np.save(f"runs/{run_dir}/train_y_ei.npy", train_y_ei.detach().cpu().numpy())

            # get gp hyperparameters as dictionary
            gp_dict = gp_ei.state_dict()
            # save gp hyperparameters to json
            torch.save(gp_dict, f"runs/{run_dir}/gp_hyperparameters_ei_iter{bo_iter}.pth")

            gp_ei = _get_gp(train_x_ei, train_y_ei)
            mll_ei = ExactMarginalLogLikelihood(gp_ei.likelihood, gp_ei)  # mll object
            fit_mll_with_adam_backup(mll_ei)  # fit mll hyperpara
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
            np.save(f"runs/{run_dir}/train_x_mes.npy", train_x_mes.detach().cpu().numpy())
            np.save(f"runs/{run_dir}/train_y_mes.npy", train_y_mes.detach().cpu().numpy())

            # get gp hyperparameters as dictionary
            gp_dict = gp_mes.state_dict()
            # save gp hyperparameters to json
            # torch.save(gp_dict, f"runs/{run_dir}/gp_hyperparameters_mes_iter{bo_iter}.pth")
            # save gp hyperparameters to tar.xz
            with tarfile.open(f"runs/{run_dir}/hyperparameters.tar.xz", "w:xz") as tar:
                gp_dict_file = io.BytesIO()
                torch.save(gp_dict, gp_dict_file)
                gp_dict_file.seek(0)
                tarinfo = tarfile.TarInfo(f"gp_hyperparameters_mes_iter{bo_iter}.pth")
                tarinfo.size = len(gp_dict_file.getbuffer())
                tar.addfile(tarinfo, gp_dict_file)

            gp_mes = _get_gp(train_x_mes, train_y_mes)
            mll_mes = ExactMarginalLogLikelihood(gp_mes.likelihood, gp_mes)
            fit_mll_with_adam_backup(mll_mes)
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
            np.save(f"runs/{run_dir}/train_x_ves.npy", train_x_ves.detach().cpu().numpy())
            np.save(f"runs/{run_dir}/train_y_ves.npy", train_y_ves.detach().cpu().numpy())

            # save k_vals and beta_vals
            np.save(f"runs/{run_dir}/k_vals.npy", np.array(k_vals))
            np.save(f"runs/{run_dir}/beta_vals.npy", np.array(beta_vals))

            # get gp hyperparameters as dictionary
            gp_dict = gp_ves.state_dict()
            # save gp hyperparameters to json
            # torch.save(gp_dict, f"runs/{run_dir}/gp_hyperparameters_ves_iter{bo_iter}.pth")
            # save gp hyperparameters to tar.xz
            with tarfile.open(f"runs/{run_dir}/hyperparameters.tar.xz", "w:xz") as tar:
                gp_dict_file = io.BytesIO()
                torch.save(gp_dict, gp_dict_file)
                gp_dict_file.seek(0)
                tarinfo = tarfile.TarInfo(f"gp_hyperparameters_ves_iter{bo_iter}.pth")
                tarinfo.size = len(gp_dict_file.getbuffer())
                tar.addfile(tarinfo, gp_dict_file)

            gp_ves = _get_gp(train_x_ves, train_y_ves)

            mll_ves = ExactMarginalLogLikelihood(gp_ves.likelihood, gp_ves)  # mll object
            fit_mll_with_adam_backup(mll_ves)  # fit mll hyperpara

            paths = robust_draw_matheron_paths(gp_ves, torch.Size([num_paths]))
            ves_model = ves_class(
                gp_ves,
                best_f=train_y_ves.max(),
                bounds=bounds,
                paths=paths,
                clamp_min=clamp_min,
                acqf_options=acqf_options,
                device=device,
            )

            _time_passed = time.time() - start_time
            print(f"Time passed: {_time_passed} seconds")
            # save the time passed, overwrite the file if it exists
            with open(f"runs/{run_dir}/time_taken.txt", "w") as file:
                file.write(str(_time_passed))
    end_time = time.time()
    seconds_passed = end_time - start_time
    print(f"Time taken: {seconds_passed} seconds")
    # save the time taken
    with open(f"runs/{run_dir}/time_taken.txt", "w") as file:
        file.write(str(seconds_passed))
