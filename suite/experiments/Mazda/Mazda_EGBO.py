import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
import csv
import uuid
from pathlib import Path
import traceback
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination

import warnings
from botorch.exceptions import InputDataWarning
warnings.filterwarnings("ignore", category=InputDataWarning)

from joblib import Parallel, delayed


# Settings
seed = 339
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Budget + batching
TOTAL_BUDGET = 1500
num_random_sample = 50

N_JOBS = 5                 # parallel exe evaluations
BATCH_SIZE = N_JOBS        # q in EGBO / q(Log)NEHVI

MC_SAMPLES = 16
GP_MAXITER = 50
EXE_TIMEOUT = 60


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATH_CON = PROJECT_ROOT / "problem_sets" / "mazda_interface" / "Info_test.xlsx"
PATH_EXE = PROJECT_ROOT / "problem_sets" / "mazda_interface" / "mazda_mop.exe"
PATH_RESULT = PROJECT_ROOT / "results" / "mazda" / "SBO_EGBO"

PATH_RESULT.mkdir(parents=True, exist_ok=True)
LOG_CSV = PATH_RESULT / f"Mazda_EGBO_seed{seed}.csv"

RUN_ROOT = PATH_RESULT / "runs" / "MAZDA_EGBO"
RUN_ROOT.mkdir(parents=True, exist_ok=True)


# Reference point (in max-space)
ref_point = torch.tensor([-1.1, 0.0], dtype=torch.float32)

# EGBO / optimizer settings
NUM_RESTARTS = 5
RAW_SAMPLES = 128
NSGA_POP_SIZE = 256


# Logging init
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "eval_id",
            "objectives",
            "objectives_maxspace",
            "is_feasible",
            "variables",
            "constraints_raw",
            "cv",
            "evaluation_time",
            "algorithm_time",
            "workdir",
        ])


def dv_range(df_path) -> list:
    """
    Retrieve the discrete domains of the decision variables from Excel.
    """
    decision_variable = pd.read_excel(df_path)
    volume_lists = []
    for _, row in decision_variable.iterrows():
        dv = row.get("Design Variable", None)
        volume_str = row.get("Discrete Volume", None)
        if pd.isna(dv) or pd.isna(volume_str):
            continue
        values = [float(v.strip()) for v in str(volume_str).split(",")]
        volume_lists.append(values)
    return volume_lists


def calc_cv_from_con_raw(con_raw_list):
    """
    Mazda: con >= 0 feasible
    Convert to cv = sum(max(0, -con))
    Then feasibility is cv <= 0
    """
    c = torch.tensor(con_raw_list, dtype=torch.float32)
    return torch.clamp(-c, min=0.0).sum().item()


def run_exe_in_dir(exe_path: str, workdir: Path, timeout=60):
    """
    Mazda exe signature: [exe_path, output_dir]
    The exe reads pop_vars_eval.txt inside output_dir and writes pop_objs_eval.txt/pop_cons_eval.txt.
    """
    subprocess.run([exe_path, str(workdir)], check=True, timeout=timeout)


def read_eval_files_in_dir(workdir: Path):
    file_dv = workdir / "pop_vars_eval.txt"
    file_obj = workdir / "pop_objs_eval.txt"
    file_con = workdir / "pop_cons_eval.txt"

    with open(file_obj, "r") as f:
        objs = [float(x) for x in f.read().split()]
    with open(file_dv, "r") as f:
        dvs = [float(x) for x in f.read().split()]
    with open(file_con, "r") as f:
        cons = [float(x) for x in f.read().split()]

    return objs, dvs, cons


def transform_obj_to_max_space(obj_original):
    """
    Original transform:
      f1_hv = obj1 - 2.0
      f2_hv = obj2 / 74.0
      y_obj = [-f1_hv, -f2_hv]  # max-space for BoTorch
    """
    f1_hv = obj_original[0] - 2.0
    f2_hv = obj_original[1] / 74.0
    y_obj = [-f1_hv, -f2_hv]
    return y_obj


def make_workdir():
    sim_id = 255 + (uuid.uuid4().int % 1_000_000)
    wd = RUN_ROOT / f"sim_{sim_id}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def eval_vars_one(vars_one, timeout=60):

    workdir = make_workdir()

    try:
        dv_file = workdir / "pop_vars_eval.txt"
        with open(dv_file, "w") as f:
            f.write("\t".join(map(str, vars_one)) + "\n")

        t0 = time.perf_counter()
        run_exe_in_dir(PATH_EXE, workdir, timeout=timeout)
        t1 = time.perf_counter()
        eval_time = t1 - t0

        obj_original, _, con_raw = read_eval_files_in_dir(workdir)

        y_obj = transform_obj_to_max_space(obj_original)
        cv = calc_cv_from_con_raw(con_raw)
        is_feasible = all(x >= 0 for x in con_raw)

        return y_obj, cv, obj_original, con_raw, is_feasible, eval_time, str(workdir)

    except Exception as e:
        print("[eval_vars_one ERROR]", repr(e))
        traceback.print_exc()

        # penalize failures
        obj_original = [1e6, 1e6]
        con_raw = [-1e6] * 54
        y_obj = [-1e6, -1e6]
        cv = 1e6
        is_feasible = False
        eval_time = None

        return y_obj, cv, obj_original, con_raw, is_feasible, eval_time, str(workdir)


def sample_random_vars(dv_ranges):
    return [rng.choice(values) for values in dv_ranges]


def build_bounds_from_dv(dv):
    lb = torch.tensor([min(v) for v in dv], dtype=torch.float32)
    ub = torch.tensor([max(v) for v in dv], dtype=torch.float32)
    rng_ = (ub - lb).clamp_min(1e-12)
    return lb, ub, rng_


def norm_X(X, lb, rng_):
    return (X - lb) / rng_


def denorm_X(Xn, lb, rng_):
    return lb + Xn * rng_


def project_continuous_to_discrete(X_cont, dv):

    out = []
    for row in X_cont:
        row_vars = []
        for j, v in enumerate(row):
            vals = dv[j]
            nearest = min(vals, key=lambda x: abs(float(x) - float(v.item())))
            row_vars.append(float(nearest))
        out.append(row_vars)
    return out


def unique_keep_order(list_of_rows):
    seen = set()
    out = []
    for row in list_of_rows:
        key = tuple(float(x) for x in row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def build_model(train_X_n, train_obj, train_cv):
    """
    2 objectives + 1 aggregated constraint violation CV
    """
    m_obj1 = SingleTaskGP(train_X_n, train_obj[:, 0:1].contiguous())
    m_obj2 = SingleTaskGP(train_X_n, train_obj[:, 1:2].contiguous())
    m_cv = SingleTaskGP(train_X_n, train_cv[:, 0:1].contiguous())

    model = ModelListGP(m_obj1, m_obj2, m_cv)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def score_candidates_with_acq(acq, candidates_n):
    scores = []
    with torch.no_grad():
        for i in range(candidates_n.shape[0]):
            val = acq(candidates_n[i].unsqueeze(0))
            scores.append(val.item())
    return torch.tensor(scores, dtype=torch.float32)


def egbo_generate_batch(
    acq,
    train_X_n,
    train_obj,
    train_cv,
    q,
    d,
):
    """
    Original EGBO structure:
      1) q(Log)NEHVI candidate batch via optimize_acqf
      2) NSGA-III candidate pool via ask/tell/ask
      3) merge candidates
      4) rerank all by acquisition
      5) take top-q in continuous normalized space
    """
    bounds = torch.stack([
        torch.zeros(d, dtype=torch.float32),
        torch.ones(d, dtype=torch.float32),
    ])


    # BO candidate batch
    qnehvi_x, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # EA candidate pool
    pareto_mask = is_non_dominated(train_obj)
    pareto_x = train_X_n[pareto_mask]
    pareto_y = -train_obj[pareto_mask]   # pymoo uses minimization
    pareto_con = train_cv[pareto_mask]

    # fallback: if not enough pareto points, use whole history
    if pareto_x.shape[0] < 2:
        pareto_x = train_X_n
        pareto_y = -train_obj
        pareto_con = train_cv

    algorithm = UNSGA3(
        pop_size=NSGA_POP_SIZE,
        ref_dirs=get_reference_directions(
            "energy",
            train_obj.shape[1],
            q,
            seed=seed,
        ),
        sampling=pareto_x.detach().cpu().numpy(),
    )

    pymooproblem = PymooProblem(
        n_var=d,
        n_obj=train_obj.shape[1],
        n_constr=1,
        xl=np.zeros(d),
        xu=np.ones(d),
    )


    algorithm.setup(
        pymooproblem,
        termination=NoTermination(),
        seed=seed
    )

    pop = algorithm.ask()
    pop.set("F", pareto_y.detach().cpu().numpy())
    pop.set("G", pareto_con.detach().cpu().numpy())
    algorithm.tell(infills=pop)

    newpop = algorithm.ask()
    nsga_x = torch.tensor(newpop.get("X"), dtype=torch.float32)


    # Merge candidates
    candidates = torch.cat([qnehvi_x, nsga_x], dim=0)

    # Acquisition rerank
    scores = score_candidates_with_acq(acq, candidates)
    idx = torch.argsort(scores, descending=True)
    best = candidates[idx[:q]]

    return best



# Main
if __name__ == "__main__":

    # Optional cleanup
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    # Load DV ranges
    dv = dv_range(PATH_CON)
    d = len(dv)
    print(f"[Info] #decision vars = {d}")

    lb, ub, rng_ = build_bounds_from_dv(dv)


    # Initial sampling (parallel)
    t_RS = time.perf_counter()

    init_vars_list = [sample_random_vars(dv) for _ in range(num_random_sample)]
    init_vars_list = unique_keep_order(init_vars_list)

    while len(init_vars_list) < num_random_sample:
        init_vars_list += [sample_random_vars(dv) for _ in range(num_random_sample)]
        init_vars_list = unique_keep_order(init_vars_list)

    init_vars_list = init_vars_list[:num_random_sample]

    init_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(eval_vars_one)(v, timeout=EXE_TIMEOUT) for v in init_vars_list
    )

    t_RE = time.perf_counter()
    print(f"[Init] done {num_random_sample} evals in {t_RE - t_RS:.3f}s")

    # Build initial train tensors
    train_X_list = []
    train_obj_list = []
    train_cv_list = []

    eval_id = 0
    for vars_one, (y_obj, cv, obj_original, con_raw, is_feasible, eval_time, workdir) in zip(init_vars_list, init_results):
        eval_id += 1
        train_X_list.append(vars_one)
        train_obj_list.append(y_obj)
        train_cv_list.append([cv])

        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                eval_id,
                obj_original,
                y_obj,
                is_feasible,
                vars_one,
                con_raw,
                cv,
                eval_time,
                None,
                workdir,
            ])

    train_X = torch.tensor(train_X_list, dtype=torch.float32)      # (n, d)
    train_obj = torch.tensor(train_obj_list, dtype=torch.float32)  # (n, 2) in max-space
    train_cv = torch.tensor(train_cv_list, dtype=torch.float32)    # (n, 1), feasible if <= 0


    # EGBO loop until TOTAL_BUDGET
    round_id = 0

    while train_X.shape[0] < TOTAL_BUDGET:
        round_id += 1
        t_alg0 = time.perf_counter()

        # Normalize current train X
        train_X_n = norm_X(train_X, lb, rng_)

        # Fit GP models
        model, mll = build_model(train_X_n, train_obj, train_cv)
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": GP_MAXITER}},
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # qLogNEHVI with aggregated constraint: cv <= 0 feasible
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_n,
            sampler=sampler,
            prune_baseline=True,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            constraints=[lambda Z: Z[..., 2]],
        )


        q_now = min(BATCH_SIZE, TOTAL_BUDGET - train_X.shape[0])

        # EGBO batch proposal in continuous normalized space
        X_next_n_cont = egbo_generate_batch(
            acq=acq,
            train_X_n=train_X_n,
            train_obj=train_obj,
            train_cv=train_cv,
            q=q_now,
            d=d,
        )

        # continuous normalized -> continuous original space
        X_next_cont = denorm_X(X_next_n_cont, lb, rng_)

        # project to nearest discrete Mazda design values
        vars_next_list = project_continuous_to_discrete(X_next_cont, dv)
        vars_next_list = unique_keep_order(vars_next_list)

        # fill if projection causes duplicates
        existing_keys = set(tuple(map(float, row)) for row in train_X.detach().cpu().numpy().tolist())
        new_vars_filtered = [row for row in vars_next_list if tuple(row) not in existing_keys]

        while len(new_vars_filtered) < q_now:
            extra = sample_random_vars(dv)
            if tuple(extra) not in existing_keys and tuple(extra) not in [tuple(x) for x in new_vars_filtered]:
                new_vars_filtered.append(extra)

        vars_next_list = new_vars_filtered[:q_now]

        t_alg1 = time.perf_counter()
        alg_time = t_alg1 - t_alg0

  
        # Parallel black-box eval for this batch
        batch_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_vars_one)(v, timeout=EXE_TIMEOUT) for v in vars_next_list
        )

        # Update train sets + log
        for vars_one, (y_obj, cv, obj_original, con_raw, is_feasible, eval_time, workdir) in zip(vars_next_list, batch_results):
            eval_id += 1

            train_X = torch.cat([train_X, torch.tensor([vars_one], dtype=torch.float32)], dim=0)
            train_obj = torch.cat([train_obj, torch.tensor([y_obj], dtype=torch.float32)], dim=0)
            train_cv = torch.cat([train_cv, torch.tensor([[cv]], dtype=torch.float32)], dim=0)

            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    eval_id,
                    obj_original,
                    y_obj,
                    is_feasible,
                    vars_one,
                    con_raw,
                    cv,
                    eval_time,
                    alg_time,
                    workdir,
                ])

        n_feas = int((train_cv.view(-1) <= 0).sum().item())
        print(
            f"[EGBO] round={round_id}  "
            f"evals={train_X.shape[0]}/{TOTAL_BUDGET}  "
            f"q={q_now}  "
            f"feasible={n_feas}  "
            f"alg_time={alg_time:.3f}s"
        )

    print(f"[Done] Total evaluations: {train_X.shape[0]}")
    print(f"[Done] Log saved: {LOG_CSV}")