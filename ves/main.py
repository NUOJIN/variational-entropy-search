from __future__ import annotations

import copy
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
from botorch.acquisition import LogExpectedImprovement, AcquisitionFunction, qKnowledgeGradient, UpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples
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
from ves.ves_gamma import VariationalEntropySearchGamma, VariationalEntropySearchGammaNew
from ves.ves_gamma_seq_k import VariationalEntropySearchGammaSeqK

if __name__ == "__main__":
    # Test VES on a trivial example (D=5)

    argparse = ArgumentParser()
    argparse.add_argument(
        "--num_paths", type=int, default=128, help="Number of paths to sample"
    )
    argparse.add_argument(
        "--num_iter", type=int, default=5, help="Number of iterations for VES"
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
    argparse.add_argument("--run_pes", type=str2bool, default=False)
    argparse.add_argument("--run_kg", type=str2bool, default=False)
    argparse.add_argument("--run_ucb", type=str2bool, default=False)
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
    argparse.add_argument(
        '--varpro', type=str2bool, default=False, help='Use Variable Projection'
    )
    argparse.add_argument(
        '--reg_lambda', type=float, default=0.0, help='Regularization parameter'
    )
    argparse.add_argument(
        '--reg_target', type=float, default=1.0, help='The value we force k to be close to'
    )
    argparse.add_argument(
        '--reg_method', type=str, default="L2", choices=["L2", "proximal"], help='Regularization method'
    )

    args = argparse.parse_args()

    num_paths = args.num_paths
    benchmark_name = args.benchmark
    clamp_min = args.clamp_min
    run_ei = args.run_old_ei
    run_log_ei = args.run_ei
    run_mes = args.run_mes
    run_pes = args.run_pes
    run_vesseq = args.run_vesseq
    run_kg = args.run_kg
    run_ucb = args.run_ucb
    init_k = args.k_init
    gp_lengthscale = args.set_lengthscale
    gp_noise = args.set_noise
    gp_outputscale = args.set_outputscale
    lengthscale_prior = args.lengthscale_prior
    stop_tolerance_coeff = args.stop_tolerance_coeff
    varpro = args.varpro
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")
    reg_lambda = args.reg_lambda
    reg_method = args.reg_method
    reg_target = args.reg_target

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

    # wait until the directory is created
    while not os.path.exists(f"runs/{run_dir}"):
        time.sleep(1)

    # save args.json to run_dir
    with open(f"runs/{run_dir}/args.json", "w") as file:
        json.dump(args_dir, file)

    if args.exponential_family:
        ves_class = VariationalEntropySearchExponential
    elif varpro:
        ves_class = VariationalEntropySearchGammaNew
    else:
        ves_class = VariationalEntropySearchGamma
    acqf_options = {
        "num_restarts": args.acqf_num_restarts,
        "raw_samples" : args.acqf_raw_samples,
        "options"     : {"sample_around_best": args.sample_around_best},
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
    if run_pes:
        train_x_pes = train_x_ves.clone()
        train_y_pes = train_y_ves.clone()
    if run_kg:
        train_x_kg = train_x_ves.clone()
        train_y_kg = train_y_ves.clone()

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
    elif run_pes:
        gp_pes = _get_gp(train_x_pes, train_y_pes)
        mll_pes = ExactMarginalLogLikelihood(gp_pes.likelihood, gp_pes)  # mll object
        fit_mll_with_adam_backup(mll_pes)  # fit mll hyperparameters
        _pes_func = qPredictiveEntropySearch
        # copy gp to move to cpu
        optimal_inputs, optimal_outputs = get_optimal_samples(
            copy.deepcopy(gp_pes).cpu(), bounds=bounds.cpu().T, num_optima=12
        )
        pes_model = _pes_func(
            gp_pes,
            optimal_inputs=optimal_inputs.to(device=device),
        )
    elif run_kg:
        gp_kg = _get_gp(train_x_kg, train_y_kg)
        mll_kg = ExactMarginalLogLikelihood(gp_kg.likelihood, gp_kg)
        fit_mll_with_adam_backup(mll_kg)  # fit mll hyperparameters
        # get the best observation
        _kg_func = qKnowledgeGradient
        kg_model = _kg_func(
            gp_kg,
            num_fantasies=64
        )
    elif run_ucb:
        gp_ucb = _get_gp(train_x_ves, train_y_ves)
        mll_ucb = ExactMarginalLogLikelihood(gp_ucb.likelihood, gp_ucb)
        fit_mll_with_adam_backup(mll_ucb)  # fit mll hyperparameters
        # get the best observation
        _ucb_func = UpperConfidenceBound
        ucb_model = _ucb_func(
            gp_ucb,
            beta=0.1,
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
            reg_lambda=reg_lambda,
            reg_target=reg_target,
        )
        k_vals = []
        beta_vals = []

    start_time = time.time()

    for bo_iter in range(args.num_bo_iter):
        print(f"+++ Iteration {bo_iter} +++")
        # Define an intial point for VES-Gamma
        X = torch.rand(1, 1, D, dtype=torch.double, device=device)


        def optimize_af_and_fit_model(
                af_model: AcquisitionFunction,
                gp_model: gpytorch.models.ExactGP,
                x_data: torch.Tensor,
                y_data: torch.Tensor,
                af_name: str
        ):
            with gpytorch.settings.cholesky_max_tries(9):
                af_candidate, acq_value = optimize_acqf(
                    af_model,
                    bounds=bounds.T,
                    q=1,  # Number of candidates to optimize for
                    num_restarts=args.acqf_num_restarts,
                    raw_samples=args.acqf_raw_samples,
                )
            x_data = torch.cat([x_data, af_candidate], dim=0)
            f_next = objective(af_candidate)
            print(
                f"{af_name}: cand={af_candidate}, acq_val={acq_value:.3e}, f_val={f_next.item():.3e}, f_max={y_data.max()}"
            )
            y_data = torch.cat([y_data, f_next.reshape(1, 1)], dim=0)
            # save the results
            np.save(
                f"runs/{run_dir}/train_x_{af_name}.npy", x_data.detach().cpu().numpy()
            )
            np.save(
                f"runs/{run_dir}/train_y_{af_name}.npy", y_data.detach().cpu().numpy()
            )

            # get gp hyperparameters as dictionary
            gp_dict = gp_model.state_dict()
            # save gp hyperparameters to json
            # torch.save(gp_dict, f"runs/{run_dir}/gp_hyperparameters_mes_iter{bo_iter}.pth")
            # save gp hyperparameters to tar.xz
            with tarfile.open(f"runs/{run_dir}/hyperparameters.tar.xz", "w:xz") as tar:
                gp_dict_file = io.BytesIO()
                torch.save(gp_dict, gp_dict_file)
                gp_dict_file.seek(0)
                tarinfo = tarfile.TarInfo(f"gp_hyperparameters_{af_name}_iter{bo_iter}.pth")
                tarinfo.size = len(gp_dict_file.getbuffer())
                tar.addfile(tarinfo, gp_dict_file)

            gp_model = _get_gp(x_data, y_data)
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)  # mll object
            fit_mll_with_adam_backup(mll)  # fit mll hyperparameters
            return gp_model, mll, x_data, y_data


        if run_ei or run_log_ei:
            gp_ei, mll_ei, train_x_ei, train_y_ei = optimize_af_and_fit_model(
                ei_model,
                gp_ei,
                train_x_ei,
                train_y_ei,
                "ei"
            )

            fit_mll_with_adam_backup(mll_ei)  # fit mll hyperpara
            _ei_func = LogExpectedImprovement if run_log_ei else LogExpectedImprovement
            ei_model = _ei_func(gp_ei, train_y_ei.max())
        elif run_mes:
            gp_mes, mll_mes, train_x_mes, train_y_mes, optimize_af_and_fit_model(
                mes_model,
                gp_mes,
                train_x_mes,
                train_y_mes,
                "mes"
            )

            gp_mes = _get_gp(train_x_mes, train_y_mes)
            mll_mes = ExactMarginalLogLikelihood(gp_mes.likelihood, gp_mes)
            fit_mll_with_adam_backup(mll_mes)
            mes_model = qMaxValueEntropy(gp_mes, candidate_set)
        elif run_vesseq:
            gp_vesseq, mll_vesseq, train_x_vesseq, train_y_vesseq = optimize_af_and_fit_model(
                vesseq_model,
                gp_vesseq,
                train_x_vesseq,
                train_y_vesseq,
                "vesseq"
            )

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
        elif run_pes:
            gp_pes, mll_pes, train_x_pes, train_y_pes = optimize_af_and_fit_model(
                pes_model,
                gp_pes,
                train_x_pes,
                train_y_pes,
                "pes"
            )

            _pes_func = qPredictiveEntropySearch
            optimal_inputs, optimal_outputs = get_optimal_samples(
                copy.deepcopy(gp_pes).cpu(), bounds=bounds.cpu().T, num_optima=12
            )
            pes_model = _pes_func(
                gp_pes,
                optimal_inputs=optimal_inputs.to(device=device),
            )
        elif run_kg:
            gp_kg, mll_kg, train_x_kg, train_y_kg = optimize_af_and_fit_model(
                kg_model,
                gp_kg,
                train_x_kg,
                train_y_kg,
                "kg"
            )

            _kg_func = qKnowledgeGradient
            kg_model = _kg_func(
                gp_kg,
                num_fantasies=64
            )
        elif run_ucb:
            gp_ucb, mll_ucb, train_x_ucb, train_y_ucb = optimize_af_and_fit_model(
                ucb_model,
                gp_ucb,
                train_x_ves,
                train_y_ves,
                "ucb"
            )

            _ucb_func = UpperConfidenceBound
            ucb_model = _ucb_func(
                gp_ucb,
                beta=0.1,
            )
        else:
            ves_candidate, v, k_val, beta_val = ves_model(
                X, num_paths=num_paths, num_iter=args.num_iter
            )
            if reg_method == "proximal":
                print(f"next k_target: {k_val}")
                reg_target = k_val
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
                reg_lambda=reg_lambda,
                reg_target=reg_target,
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
