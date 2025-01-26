from __future__ import annotations

import io
import json
import math
import os
import tarfile
import time
from argparse import ArgumentParser
from functools import partial
from zlib import adler32

import gpytorch
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.sampling import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood

from ves.util import (
    str2bool,
    get_gp,
    fit_mll_with_adam_backup,
    robust_draw_matheron_paths,
    get_objective,
    AVAILABLE_BENCHMARKS,
    init_samples,
)
from ves.ves_exponential import VariationalEntropySearchExponential
from ves.ves_gamma import VariationalEntropySearchGamma
from ves.ves_gamma_seq_k import VariationalEntropySearchGammaSeqK

if __name__ == "__main__":
    # Test VES on a trivial example (D=5)

    argparse = ArgumentParser()
    argparse.add_argument(
        "--num_paths", type=int, default=64, help="Number of paths to sample"
    )
    argparse.add_argument(
        "--num_iter", type=int, default=50, help="Number of iterations for VES"
    )
    argparse.add_argument("--num_bo_iter", type=int, default=500)
    argparse.add_argument("--n_init", type=int, default=20)
    argparse.add_argument("--k_init", type=float, default=None)
    argparse.add_argument("--clamp_min", type=float, default=1e-10)
    argparse.add_argument("--acqf_raw_samples", type=int, default=512)
    argparse.add_argument("--acqf_num_restarts", type=int, default=5)
    argparse.add_argument("--sample_around_best", type=str2bool, default=True)
    argparse.add_argument("--run_ei", type=str2bool, default=False)
    argparse.add_argument("--run_old_ei", type=str2bool, default=False)
    argparse.add_argument("--run_mes", type=str2bool, default=False)
    argparse.add_argument("--run_vesseq", type=str2bool, default=False)
    argparse.add_argument(
        "--decay_target",
        type=int,
        default=None,
        help="The number of iterations to reach the final k (num_bo_iter if None)",
    )
    argparse.add_argument(
        "--k_target", type=float, default=0.5, help="The final k value"
    )
    argparse.add_argument("--exponential_family", type=str2bool, default=False)
    argparse.add_argument("--set_lengthscale", type=float, default=None)
    argparse.add_argument("--set_noise", type=float, default=None)
    argparse.add_argument("--set_outputscale", type=float, default=None)
    argparse.add_argument(
        "--lengthscale_prior", choices=["bounce", "vbo"], default="bounce"
    )
    argparse.add_argument("--stop_tolerance_coeff", type=float, default=1e-5)
    argparse.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    argparse.add_argument(
        "--benchmark", type=str, choices=AVAILABLE_BENCHMARKS, required=True
    )

    args = argparse.parse_args()

    num_paths = args.num_paths
    benchmark_name = args.benchmark
    clamp_min = args.clamp_min
    run_ei = args.run_old_ei
    run_log_ei = args.run_ei
    run_mes = args.run_mes
    run_vesseq = args.run_vesseq
    init_k = args.k_init
    gp_lengthscale = args.set_lengthscale
    gp_noise = args.set_noise
    gp_outputscale = args.set_outputscale
    lengthscale_prior = args.lengthscale_prior
    stop_tolerance_coeff = args.stop_tolerance_coeff
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    # check that at most one of run_ei, run_log_ei, run_mes, run_vesseq is True
    assert (
        sum([run_ei, run_log_ei, run_mes, run_vesseq]) <= 1
    ), "At most one of run_ei, run_log_ei, run_mes, run_vesseq can be True"

    # Define the objective function
    objective, D = get_objective(benchmark_name=args.benchmark, device=device)

    # if init_k is None, use D as the default value
    if init_k is None:
        print("Using D as the default value for k")
        init_k = D

    args_dir = vars(args)
    # calculate run_dir hash with adler32
    run_dir = f"{adler32(json.dumps(args_dir).encode())}"
    timestamp_ms = int(time.time_ns() / 1e6)
    run_dir = f"{timestamp_ms}_{run_dir}"
    if "SLURM_ARRAY_TASK_ID" in os.environ:
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
        "raw_samples": args.acqf_raw_samples,
        "options": {"sample_around_best": args.sample_around_best},
    }

    n_init = args.n_init
    # train_x_ves = torch.rand(n_init, D, dtype=torch.double, device=device)
    train_x_ves = init_samples(
        n_init=n_init, dim=D, seed=os.environ.get("SLURM_ARRAY_TASK_ID", None)
    ).to(dtype=torch.double, device=device)
    train_y_ves = (
        torch.Tensor([objective(x) for x in train_x_ves])
        .unsqueeze(1)
        .to(dtype=torch.double, device=device)
    )

    if run_ei or run_log_ei:
        train_x_ei = train_x_ves.clone()
        train_y_ei = train_y_ves.clone()
    if run_mes:
        train_x_mes = train_x_ves.clone()
        train_y_mes = train_y_ves.clone()
    if run_vesseq:
        train_x_vesseq = train_x_ves.clone()
        train_y_vesseq = train_y_ves.clone()

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

    if run_ei or run_log_ei:
        gp_ei = _get_gp(train_x_ei, train_y_ei)
        mll_ei = ExactMarginalLogLikelihood(gp_ei.likelihood, gp_ei)  # mll object
        fit_mll_with_adam_backup(mll_ei)  # fit mll hyperparameters
        _ei_func = LogExpectedImprovement if run_log_ei else LogExpectedImprovement
        ei_model = _ei_func(gp_ei, train_y_ei.max())
    elif run_mes:
        gp_mes = _get_gp(train_x_mes, train_y_mes)
        mll_mes = ExactMarginalLogLikelihood(gp_mes.likelihood, gp_mes)  # mll object
        fit_mll_with_adam_backup(mll_mes)  # fit mll hyperparameters
        candidate_set = (
            SobolEngine(dimension=bounds.shape[0], scramble=True)
            .draw(2048)
            .to(device=device)
        )
        mes_model = qMaxValueEntropy(gp_mes, candidate_set)
    elif run_vesseq:
        gp_vesseq = _get_gp(train_x_vesseq, train_y_vesseq)
        mll_vesseq = ExactMarginalLogLikelihood(
            gp_vesseq.likelihood, gp_vesseq
        )  # mll object
        fit_mll_with_adam_backup(mll_vesseq)  # fit mll hyperparameters
        vesseq_model = VariationalEntropySearchGammaSeqK(
            gp_vesseq,
            best_f=train_y_vesseq.max(),
            num_paths=num_paths,
            clamp_min=clamp_min,
            k=init_k,
            bounds=bounds,
            device=device,
        )
    else:  # run VES
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
        if run_ei or run_log_ei:
            with gpytorch.settings.cholesky_max_tries(9):
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
            # torch.save(gp_dict, f"runs/{run_dir}/gp_hyperparameters_ei_iter{bo_iter}.pth")
            # save gp hyperparameters to tar.xz
            with tarfile.open(f"runs/{run_dir}/hyperparameters.tar.xz", "w:xz") as tar:
                gp_dict_file = io.BytesIO()
                torch.save(gp_dict, gp_dict_file)
                gp_dict_file.seek(0)
                tarinfo = tarfile.TarInfo(f"gp_hyperparameters_ei_iter{bo_iter}.pth")
                tarinfo.size = len(gp_dict_file.getbuffer())
                tar.addfile(tarinfo, gp_dict_file)

            gp_ei = _get_gp(train_x_ei, train_y_ei)
            mll_ei = ExactMarginalLogLikelihood(gp_ei.likelihood, gp_ei)  # mll object
            fit_mll_with_adam_backup(mll_ei)  # fit mll hyperpara
            _ei_func = LogExpectedImprovement if run_log_ei else LogExpectedImprovement
            ei_model = _ei_func(gp_ei, train_y_ei.max())
        elif run_mes:
            with gpytorch.settings.cholesky_max_tries(9):
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
            np.save(
                f"runs/{run_dir}/train_x_mes.npy", train_x_mes.detach().cpu().numpy()
            )
            np.save(
                f"runs/{run_dir}/train_y_mes.npy", train_y_mes.detach().cpu().numpy()
            )

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
        elif run_vesseq:
            with gpytorch.settings.cholesky_max_tries(9):
                vesseq_candidate, acq_value = optimize_acqf(
                    vesseq_model,
                    bounds=bounds.T,
                    q=1,  # Number of candidates to optimize for
                    num_restarts=args.acqf_num_restarts,
                    raw_samples=args.acqf_raw_samples,
                )
            train_x_vesseq = torch.cat([train_x_vesseq, vesseq_candidate], dim=0)
            f_vesseq = objective(vesseq_candidate)
            print(
                f"VESSeq: cand={vesseq_candidate}, acq_val={acq_value:.3e}, f_val={f_vesseq.item():.3e}, f_max={train_y_vesseq.max()}"
            )
            train_y_vesseq = torch.cat([train_y_vesseq, f_vesseq.reshape(1, 1)], dim=0)
            # save the results
            np.save(
                f"runs/{run_dir}/train_x_vesseq.npy",
                train_x_vesseq.detach().cpu().numpy(),
            )
            np.save(
                f"runs/{run_dir}/train_y_vesseq.npy",
                train_y_vesseq.detach().cpu().numpy(),
            )

            # get gp hyperparameters as dictionary
            gp_dict = gp_vesseq.state_dict()
            # save gp hyperparameters to json
            # torch.save(gp_dict, f"runs/{run_dir}/gp_hyperparameters_mes_iter{bo_iter}.pth")
            # save gp hyperparameters to tar.xz
            with tarfile.open(f"runs/{run_dir}/hyperparameters.tar.xz", "w:xz") as tar:
                gp_dict_file = io.BytesIO()
                torch.save(gp_dict, gp_dict_file)
                gp_dict_file.seek(0)
                tarinfo = tarfile.TarInfo(
                    f"gp_hyperparameters_vesseq_iter{bo_iter}.pth"
                )
                tarinfo.size = len(gp_dict_file.getbuffer())
                tar.addfile(tarinfo, gp_dict_file)

            gp_vesseq = _get_gp(train_x_vesseq, train_y_vesseq)
            mll_vesseq = ExactMarginalLogLikelihood(gp_vesseq.likelihood, gp_vesseq)
            fit_mll_with_adam_backup(mll_vesseq)
            k_target = args.k_target
            k_plus = init_k - k_target
            decay_target = (
                args.decay_target if args.decay_target is not None else args.num_bo_iter
            )
            # set up the decay rate so that the final k is 1.05
            decay_rate = (math.log(k_plus) - math.log(0.05)) / decay_target
            k_next = (init_k - k_target) * math.exp(-decay_rate * bo_iter) + k_target
            print(f"Next k: {k_next}")
            vesseq_model = VariationalEntropySearchGammaSeqK(
                gp_vesseq,
                best_f=train_y_vesseq.max(),
                num_paths=num_paths,
                clamp_min=clamp_min,
                # k=init_k / np.log(2 + bo_iter),
                k=k_next,
                bounds=bounds,
                device=device,
            )
        else:
            ves_candidate, v, k_val, beta_val = ves_model(
                X, num_paths=num_paths, num_iter=args.num_iter
            )
            k_vals.append(k_val)
            beta_vals.append(beta_val)
            train_x_ves = torch.cat([train_x_ves, ves_candidate], dim=0)
            f_ves = objective(ves_candidate)
            print(
                f"VES: cand={ves_candidate}, acq_val={v:.3e}, f_val={f_ves.item():.3e}, f_max={train_y_ves.max()}"
            )
            train_y_ves = torch.cat([train_y_ves, f_ves.reshape(1, 1)], dim=0)
            # save the results
            np.save(
                f"runs/{run_dir}/train_x_ves.npy", train_x_ves.detach().cpu().numpy()
            )
            np.save(
                f"runs/{run_dir}/train_y_ves.npy", train_y_ves.detach().cpu().numpy()
            )

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

            mll_ves = ExactMarginalLogLikelihood(
                gp_ves.likelihood, gp_ves
            )  # mll object
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
