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
seed = 332
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Budget + batching
TOTAL_BUDGET = 1000
num_random_sample = 50  # （保留但不再使用）

N_JOBS = 5                 # parallel exe evaluations
BATCH_SIZE = N_JOBS        # q in qLogNEHVI (parallel batch)
K = 500                    # number of discrete candidates per BO round (increase if needed)

MC_SAMPLES = 16
GP_MAXITER = 50            # GP fitting maxiter
EXE_TIMEOUT = 60

# Paths
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx"
PATH_EXE = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EHVI_PAR"

os.makedirs(PATH_RESULT, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_qLogNEHVI_seed{seed}.csv")

# NSGA2 CSV (seed=331)
NSGA2_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2"
NSGA2_CSV = os.path.join(NSGA2_RESULT, f"Mazda_NSGA2_seed{seed}.csv")

# Per-eval isolated workdirs (parallel-safe)
RUN_ROOT = Path(PATH_RESULT) / "runs" / "MAZDA_SBO"
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
            "constraints",
            "evaluation_time",
            "algorithm_time",
            # "workdir"
        ])


# =========================
# Helpers (原封不动保留你的结构)
# =========================
def dv_range(df_path) -> list:
    """
    Retrieve the domain of the decision variables from Excel.
    """
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
    Your original transform:
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


# =========================
# Main
# =========================
if __name__ == "__main__":

    # clean runs each execution (和你原来一致)
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

    # ============================================================
    # ✅ Initial sampling：从 NSGA2 CSV 读取可行解作为初始点
    # ============================================================
    df_nsga2 = pd.read_csv(NSGA2_CSV, delimiter=";")
    df_feas = df_nsga2[df_nsga2["is_feasible"] == True]

    if len(df_feas) == 0:
        raise RuntimeError("NSGA2 CSV 中没有可行解（is_feasible==True）")

    print(f"[Init] Feasible points from NSGA2: {len(df_feas)}")

    train_X_list = []
    train_obj_list = []
    train_cv_list = []

    eval_id = 0

    for _, row in df_feas.iterrows():
        # NSGA2 CSV 里保存的是字符串形式的 list，这里按你原来的风格用 eval 解析
        vars_one = eval(row["variables"])
        obj_original = eval(row["objectives"])
        con_raw = eval(row["constraints"])

        y_obj = transform_obj_to_max_space(obj_original)
        cv = calc_cv_from_con_raw(con_raw)

        train_X_list.append(vars_one)
        train_obj_list.append(y_obj)
        train_cv_list.append([cv])

        eval_id += 1

        # 把这些初始点也写入 qLogNEHVI 的日志（algorithm_time=None）
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                eval_id,
                obj_original,
                True,
                vars_one,
                con_raw,
                row["evaluation_time"] if "evaluation_time" in df_feas.columns else None,
                None
            ])

    train_X = torch.tensor(train_X_list, dtype=torch.float32)          # (n, d)
    train_obj = torch.tensor(train_obj_list, dtype=torch.float32)      # (n, 2) max-space
    train_cv  = torch.tensor(train_cv_list, dtype=torch.float32)       # (n, 1)

    # -------------------------
    # BO loop until TOTAL_BUDGET（完全保持你的原逻辑）
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
        for vars_one, (y_obj, cv, obj_original, con_raw, is_feasible, eval_time, workdir) in zip(vars_next_list, batch_results):
            eval_id += 1

            train_X = torch.cat([train_X, torch.tensor([vars_one], dtype=torch.float32)], dim=0)
            train_obj = torch.cat([train_obj, torch.tensor([y_obj], dtype=torch.float32)], dim=0)
            train_cv  = torch.cat([train_cv,  torch.tensor([[cv]], dtype=torch.float32)], dim=0)

            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([eval_id, obj_original, is_feasible, vars_one, con_raw, eval_time, alg_time, workdir])

        print(f"[BO] evals={train_X.shape[0]}/{TOTAL_BUDGET}  (last batch q={q_now}, alg_time={alg_time:.3f}s)")

    print(f"[Done] Total evaluations: {train_X.shape[0]}")
    print(f"[Done] Log saved: {LOG_CSV}")