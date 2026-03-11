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
from botorch.optim import optimize_acqf_discrete

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
seed = 331
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Budget + batching
TOTAL_BUDGET = 1000
num_random_sample = 50

N_JOBS = 5                 # parallel exe evaluations
BATCH_SIZE = N_JOBS        # q in qLogNEHVI (batch size)
K = 200                    # number of discrete candidates per BO round

MC_SAMPLES = 16
GP_MAXITER = 50            # GP fitting maxiter
EXE_TIMEOUT = 60

# Paths
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx"
PATH_EXE = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EHVI_PAR_54CONS"

os.makedirs(PATH_RESULT, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_qLogNEHVI_54CONS_seed{seed}.csv")

# Per-eval isolated workdirs (parallel-safe)
RUN_ROOT = Path(PATH_RESULT) / "runs" / "MAZDA_SBO"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# Reference point (in max-space) for 2 objectives
ref_point = torch.tensor([-1.1, 0.0], dtype=torch.float32)

# Mazda constraints count
N_CONS = 54

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
        ])

# =========================
# Helpers
# =========================
def dv_range(df_path) -> list:
    """Retrieve the domain of the decision variables from Excel."""
    dicision_variable = pd.read_excel(df_path)
    volume_lists = []
    for _, row in dicision_variable.iterrows():
        dv = row.get("Design Variable", None)
        volume_str = row.get("Discrete Volume", None)
        if pd.isna(dv) or pd.isna(volume_str):
            continue
        values = [float(v.strip()) for v in str(volume_str).split(",")]
        volume_lists.append(values)
    return volume_lists

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
    Your original transform:
      f1_hv = obj1 - 2.0
      f2_hv = obj2 / 74.0
      y_obj = [-f1_hv, -f2_hv]  # max-space for BoTorch
    """
    f1_hv = obj_original[0] - 2.0
    f2_hv = obj_original[1] / 74.0
    return [-f1_hv, -f2_hv]

def raw_cons_to_botorch_c(cons_raw):
    """
    Mazda feasibility: cons_raw[j] >= 0 for all j.
    BoTorch constraints expect c_j(x) <= 0 feasible.
    So define: c_j(x) = -cons_raw[j].
    """
    if len(cons_raw) != N_CONS:
        raise ValueError(f"Expected {N_CONS} constraints, got {len(cons_raw)}")
    return [-float(v) for v in cons_raw]  # feasible => <= 0

def make_workdir():
    sim_id = 255 + (uuid.uuid4().int % 1_000_000)
    wd = RUN_ROOT / f"sim_{sim_id}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd

def eval_vars_one(vars_one, timeout=60):
    """
    One black-box evaluation in an isolated workdir.

    Returns:
      y_obj_max (len=2),
      c_vec (len=54) where feasible iff all <= 0,
      obj_original (len=2),
      con_raw (len=54) where feasible iff all >= 0,
      is_feasible (bool),
      eval_time (float|None),
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
        c_vec = raw_cons_to_botorch_c(con_raw)
        is_feasible = all(v >= 0 for v in con_raw)

        return y_obj, c_vec, obj_original, con_raw, is_feasible, eval_time

    except Exception as e:
        print("[eval_vars_one ERROR]", repr(e))
        traceback.print_exc()

        # penalize (and mark infeasible)
        obj_original = [1e6, 1e6]
        con_raw = [-1e6] * N_CONS                   # infeasible in Mazda form
        y_obj = [-1e6, -1e6]
        c_vec = [1e6] * N_CONS                      # infeasible in BoTorch form (positive => >0)
        is_feasible = False
        eval_time = None

        return y_obj, c_vec, obj_original, con_raw, is_feasible, eval_time

def sample_random_vars(dv_ranges):
    return [rng.choice(values) for values in dv_ranges]

def make_constraints_list(n_cons: int):
    """
    Create list of constraint callables for BoTorch:
      Model outputs order:
        Z[..., 0] -> obj1
        Z[..., 1] -> obj2
        Z[..., 2 + j] -> constraint j (c_j), feasible if <= 0
    """
    # Important: use closure factory to avoid late-binding bug
    return [(lambda j: (lambda Z: Z[..., 2 + j]))(j) for j in range(n_cons)]

# =========================
# Main
# =========================
if __name__ == "__main__":

    # Clean runs each execution (comment out if you want to keep history)
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

    # -------------------------
    # Initial sampling (parallel)
    # -------------------------
    t_RS = time.perf_counter()

    init_vars_list = [sample_random_vars(dv) for _ in range(num_random_sample)]
    init_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(eval_vars_one)(v, timeout=EXE_TIMEOUT) for v in init_vars_list
    )

    t_RE = time.perf_counter()
    print(f"[Init] done {num_random_sample} evals in {t_RE - t_RS:.3f}s")

    # Build initial train tensors
    train_X_list = []
    train_obj_list = []
    train_con_list = []

    eval_id = 0
    for vars_one, (y_obj, c_vec, obj_original, con_raw, is_feasible, eval_time) in zip(init_vars_list, init_results):
        eval_id += 1
        train_X_list.append(vars_one)
        train_obj_list.append(y_obj)       # (2,)
        train_con_list.append(c_vec)       # (54,)

        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([eval_id, obj_original, is_feasible, vars_one, con_raw, eval_time, None])

    train_X = torch.tensor(train_X_list, dtype=torch.float32)          # (n, d)
    train_obj = torch.tensor(train_obj_list, dtype=torch.float32)      # (n, 2) max-space
    train_con = torch.tensor(train_con_list, dtype=torch.float32)      # (n, 54) botorch form: feasible if <= 0

    # Prebuild constraints list once
    constraints_list = make_constraints_list(N_CONS)

    # -------------------------
    # BO loop until TOTAL_BUDGET
    # -------------------------
    while train_X.shape[0] < TOTAL_BUDGET:
        t_alg0 = time.perf_counter()

        # Re-normalize
        train_X_n = norm_X(train_X)

        # Cold start GP models: 2 objs + 54 constraints => total 56 GPs
        m_obj1 = SingleTaskGP(train_X_n, train_obj[:, 0:1].contiguous())
        m_obj2 = SingleTaskGP(train_X_n, train_obj[:, 1:2].contiguous())

        m_cons = [
            SingleTaskGP(train_X_n, train_con[:, j:j+1].contiguous())
            for j in range(N_CONS)
        ]

        model = ModelListGP(m_obj1, m_obj2, *m_cons)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": GP_MAXITER}},
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # qLogNEHVI with 54 constraints: feasible if all c_j(x) <= 0
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_n,
            sampler=sampler,
            prune_baseline=True,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            constraints=constraints_list,
        )

        # Discrete candidate set
        X_cand_list = [sample_random_vars(dv) for _ in range(K)]
        X_cand = torch.tensor(X_cand_list, dtype=torch.float32)
        X_cand_n = norm_X(X_cand)

        # Select q points
        q_now = min(BATCH_SIZE, TOTAL_BUDGET - train_X.shape[0])
        X_next_n, _ = optimize_acqf_discrete(
            acq_function=acq,
            choices=X_cand_n,
            q=q_now,
        )

        # Map each row in X_next_n back to a row in X_cand (nearest neighbor)
        idxs = torch.cdist(X_cand_n, X_next_n).argmin(dim=0)  # (q,)
        X_next = X_cand[idxs, :]                              # (q, d)
        vars_next_list = [X_next[j].tolist() for j in range(X_next.shape[0])]

        t_alg1 = time.perf_counter()
        alg_time = t_alg1 - t_alg0

        # Parallel black-box eval for the batch
        batch_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_vars_one)(v, timeout=EXE_TIMEOUT) for v in vars_next_list
        )

        # Update train sets + log
        for vars_one, (y_obj, c_vec, obj_original, con_raw, is_feasible, eval_time) in zip(vars_next_list, batch_results):
            eval_id += 1

            train_X = torch.cat([train_X, torch.tensor([vars_one], dtype=torch.float32)], dim=0)
            train_obj = torch.cat([train_obj, torch.tensor([y_obj], dtype=torch.float32)], dim=0)
            train_con = torch.cat([train_con, torch.tensor([c_vec], dtype=torch.float32)], dim=0)

            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([eval_id, obj_original, is_feasible, vars_one, con_raw, eval_time, alg_time])

        print(f"[BO] evals={train_X.shape[0]}/{TOTAL_BUDGET}  (last batch q={q_now}, alg_time={alg_time:.3f}s)")

    print(f"[Done] Total evaluations: {train_X.shape[0]}")
    print(f"[Done] Log saved: {LOG_CSV}")