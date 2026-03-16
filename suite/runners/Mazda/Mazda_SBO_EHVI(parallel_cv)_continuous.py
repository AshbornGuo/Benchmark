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

import warnings
from botorch.exceptions import InputDataWarning
warnings.filterwarnings("ignore", category=InputDataWarning)

from joblib import Parallel, delayed

# =========================
# User Settings
# =========================
seed = 333
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Budget + batching
TOTAL_BUDGET = 1500
num_random_sample = 50

N_JOBS = 5                 # parallel exe evaluations
BATCH_SIZE = N_JOBS        # q in qLogNEHVI (parallel batch)

MC_SAMPLES = 16
GP_MAXITER = 50            # GP fitting maxiter
EXE_TIMEOUT = 60

# Continuous optimize_acqf settings
NUM_RESTARTS = 5
RAW_SAMPLES = 128
ACQ_MAXITER = 100

# Paths
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx"
PATH_EXE = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EHVI"

os.makedirs(PATH_RESULT, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_qLogNEHVI_seed{seed}.csv")

# Per-eval isolated workdirs (parallel-safe)
RUN_ROOT = Path(PATH_RESULT) / "runs" / "MAZDA_SBO_CONT"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# Reference point (in max-space)
ref_point = torch.tensor([-1.1, 0.0], dtype=torch.float32)

# =========================
# Logging init
# =========================
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "eval_id",
            "objectives",
            "is_feasible",
            "variables",
            "constraints_raw",
            "evaluation_time",
            "algorithm_time",
            "workdir",
        ])

# =========================
# Helpers
# =========================
def dv_range(df_path) -> list:
    """
    Retrieve the domain of the decision variables from Excel.
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
    cv = sum(max(0, -con))
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
    file_dv  = workdir / "pop_vars_eval.txt"
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
    """
    One black-box evaluation in an isolated workdir.
    Returns:
      y_obj (max-space, len=2),
      cv (float),
      obj_original (len=2),
      con_raw (len=54),
      is_feasible (bool),
      eval_time (float|None),
      workdir (str)
    """
    workdir = make_workdir()

    try:
        # write inputs in workdir
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

        # penalize
        obj_original = [1e6, 1e6]
        con_raw = [-1e6] * 54
        y_obj = [-1e6, -1e6]
        cv = 1e6
        is_feasible = False
        eval_time = None

        return y_obj, cv, obj_original, con_raw, is_feasible, eval_time, str(workdir)


def sample_random_vars(dv_ranges):
    return [rng.choice(values) for values in dv_ranges]


def unique_keep_order(list_of_rows):
    seen = set()
    out = []
    for row in list_of_rows:
        key = tuple(float(x) for x in row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def project_continuous_to_discrete(X_cont: torch.Tensor, dv: list):
    """
    Project continuous candidates to nearest discrete values for each dimension.
    X_cont: (q, d)
    returns list of discrete rows
    """
    out = []
    for row in X_cont:
        row_vars = []
        for j, v in enumerate(row):
            vals = dv[j]
            nearest = min(vals, key=lambda x: abs(float(x) - float(v.item())))
            row_vars.append(float(nearest))
        out.append(row_vars)
    return out


# =========================
# Main
# =========================
if __name__ == "__main__":

    # Optional: clean runs each execution
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    # Load DV ranges
    dv = dv_range(PATH_CON)
    d = len(dv)
    print(f"[Info] #decision vars = {d}")

    # Bounds for normalization
    lb = torch.tensor([min(v) for v in dv], dtype=torch.float32)
    ub = torch.tensor([max(v) for v in dv], dtype=torch.float32)
    rng_ = (ub - lb).clamp_min(1e-12)

    def norm_X(X):
        return (X - lb) / rng_

    def unnorm_X(Xn):
        return Xn * rng_ + lb

    # -------------------------
    # Initial sampling (parallel)
    # -------------------------
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
            writer.writerow([eval_id, obj_original, is_feasible, vars_one, con_raw, eval_time, None, workdir])

    train_X = torch.tensor(train_X_list, dtype=torch.float32)          # (n, d)
    train_obj = torch.tensor(train_obj_list, dtype=torch.float32)      # (n, 2) max-space
    train_cv  = torch.tensor(train_cv_list, dtype=torch.float32)       # (n, 1)

    # -------------------------
    # BO loop until TOTAL_BUDGET
    # -------------------------
    while train_X.shape[0] < TOTAL_BUDGET:
        t_alg0 = time.perf_counter()

        # Re-normalize
        train_X_n = norm_X(train_X)

        # Cold start GP models: 2 objs + 1 cv
        m_obj1 = SingleTaskGP(train_X_n, train_obj[:, 0:1].contiguous())
        m_obj2 = SingleTaskGP(train_X_n, train_obj[:, 1:2].contiguous())
        m_cv   = SingleTaskGP(train_X_n, train_cv[:, 0:1].contiguous())

        model = ModelListGP(m_obj1, m_obj2, m_cv)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": GP_MAXITER}},
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # qLogNEHVI with constraint: cv <= 0 feasible
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_n,
            sampler=sampler,
            prune_baseline=True,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            constraints=[lambda Z: Z[..., 2]],   # feasible if <= 0
        )

        # Select q points in continuous normalized space
        q_now = min(BATCH_SIZE, TOTAL_BUDGET - train_X.shape[0])

        bounds = torch.stack([
            torch.zeros(d, dtype=torch.float32),
            torch.ones(d, dtype=torch.float32)
        ])

        X_next_n, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q_now,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"maxiter": ACQ_MAXITER},
        )

        # Continuous normalized -> continuous original
        X_next_cont = unnorm_X(X_next_n)

        # Project to nearest discrete values
        vars_next_list = project_continuous_to_discrete(X_next_cont, dv)
        vars_next_list = unique_keep_order(vars_next_list)

        # Remove already evaluated points if possible
        existing_keys = set(tuple(map(float, row)) for row in train_X.detach().cpu().numpy().tolist())
        new_vars_filtered = [row for row in vars_next_list if tuple(map(float, row)) not in existing_keys]

        # Fill if projection causes duplicates or collisions
        while len(new_vars_filtered) < q_now:
            extra = sample_random_vars(dv)
            key = tuple(map(float, extra))
            if key not in existing_keys and key not in [tuple(map(float, x)) for x in new_vars_filtered]:
                new_vars_filtered.append(extra)

        vars_next_list = new_vars_filtered[:q_now]

        t_alg1 = time.perf_counter()
        alg_time = t_alg1 - t_alg0

        # Parallel black-box eval for the batch
        batch_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_vars_one)(v, timeout=EXE_TIMEOUT) for v in vars_next_list
        )

        # Update train sets + log
        for vars_one, (y_obj, cv, obj_original, con_raw, is_feasible, eval_time, workdir) in zip(vars_next_list, batch_results):
            eval_id += 1

            train_X = torch.cat([train_X, torch.tensor([vars_one], dtype=torch.float32)], dim=0)
            train_obj = torch.cat([train_obj, torch.tensor([y_obj], dtype=torch.float32)], dim=0)
            train_cv  = torch.cat([train_cv,  torch.tensor([[cv]], dtype=torch.float32)], dim=0)

            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([eval_id, obj_original, is_feasible, vars_one, con_raw, eval_time, alg_time, workdir])

        print(f"[BO-continuous] evals={train_X.shape[0]}/{TOTAL_BUDGET}  (last batch q={q_now}, alg_time={alg_time:.3f}s)")

    print(f"[Done] Total evaluations: {train_X.shape[0]}")
    print(f"[Done] Log saved: {LOG_CSV}")










    