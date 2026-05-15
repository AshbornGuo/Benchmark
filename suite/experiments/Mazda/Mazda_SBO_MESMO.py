import os
import time
import shutil
import subprocess
from pathlib import Path
import csv
import uuid
import numpy as np
import pandas as pd

import torch
from joblib import Parallel, delayed

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.optim import optimize_acqf
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated

from torch.distributions import Normal

import warnings
from botorch.exceptions import InputDataWarning
warnings.filterwarnings("ignore", category=InputDataWarning)


seed = 339
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

num_eval = 1500
n_init = 50
n_jobs = 5
mc_samples = 16
dtype = torch.double
device = torch.device("cpu")

num_restarts = 5
raw_samples = 128
maxiter = 100

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATH_CON = PROJECT_ROOT / "problem_sets" / "mazda_interface" / "Info_test.xlsx"
PATH_EXE = PROJECT_ROOT / "problem_sets" / "mazda_interface" / "mazda_mop.exe"
PATH_RESULT = PROJECT_ROOT / "results" / "mazda" / "MESMO"

PATH_RESULT.mkdir(parents=True, exist_ok=True)
LOG_CSV = PATH_RESULT / f"Mazda_MESMO_seed{seed}.csv"

RUN_ROOT = PATH_RESULT / "runs" / "MAZDA_SBO_CONT"
RUN_ROOT.mkdir(parents=True, exist_ok=True)



def dv_range(df_path) -> list:
    decision_variable = pd.read_excel(df_path)
    volume_lists = []

    for _, row in decision_variable.iterrows():
        dv = row["Design Variable"]
        volume_str = row["Discrete Volume"]

        if pd.isna(dv) or pd.isna(volume_str):
            continue

        values = [float(v.strip()) for v in str(volume_str).split(",")]
        volume_lists.append(values)

    return volume_lists


DV_RANGES = dv_range(PATH_CON)
dim = len(DV_RANGES)

lb = torch.tensor([min(v) for v in DV_RANGES], dtype=dtype, device=device)
ub = torch.tensor([max(v) for v in DV_RANGES], dtype=dtype, device=device)
rng_ = (ub - lb).clamp_min(1e-12)


def run_exe(exe_path, input_txt, output_dir, timeout=60):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(input_txt, os.path.join(output_dir, "pop_vars_eval.txt"))
    subprocess.run([str(exe_path), str(output_dir)], check=True, timeout=timeout)


def read_eval_files(path_result):
    file_dv = os.path.join(path_result, "pop_vars_eval.txt")
    file_obj = os.path.join(path_result, "pop_objs_eval.txt")
    file_con = os.path.join(path_result, "pop_cons_eval.txt")

    with open(file_obj, "r") as f:
        objs = [float(x) for x in f.read().split()]
    with open(file_dv, "r") as f:
        dvs = [float(x) for x in f.read().split()]
    with open(file_con, "r") as f:
        cons = [float(x) for x in f.read().split()]

    return objs, dvs, cons


def calc_cv_from_con_raw(con_raw_list):
    c = torch.tensor(con_raw_list, dtype=torch.double)
    return torch.clamp(-c, min=0.0).sum().item()



# evaluation
def run_one_sim(sim_id: int, vector):
    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    input_txt = workdir / "input_vars.txt"

    with open(input_txt, "w") as f:
        f.write("\t".join(map(str, vector)) + "\n")

    t0 = time.perf_counter()
    run_exe(PATH_EXE, str(input_txt), str(workdir))
    obj_original, dvs, con_raw = read_eval_files(str(workdir))
    t1 = time.perf_counter()

    eval_time = t1 - t0
    return obj_original, con_raw, eval_time


def eval_one_x(x_np: np.ndarray):
    sim_id = 255 + (uuid.uuid4().int % 1_000_000)
    x_list = x_np.tolist()

    obj_original, con_raw, eval_time = run_one_sim(sim_id, x_list)

    f1 = float(obj_original[0])
    f2 = float(obj_original[1])

    f1_hv = f1 - 2.0
    f2_hv = f2 / 74.0

    y = np.array([-f1_hv, -f2_hv], dtype=np.double)

    cv = float(calc_cv_from_con_raw(con_raw))
    is_feasible = (cv <= 1e-12)

    return {
        "sim_id": sim_id,
        "x": x_list,
        "objs_original": [f1, f2],
        "con_raw": list(map(float, con_raw)),
        "cv": cv,
        "is_feasible": is_feasible,
        "y_max": y.tolist(),
        "eval_time": float(eval_time),
    }


def eval_batch(X_np: np.ndarray, n_jobs: int):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(eval_one_x)(X_np[i]) for i in range(X_np.shape[0])
    )
    return results



# Normalization 
def norm_X(X: torch.Tensor) -> torch.Tensor:
    return (X - lb) / rng_


def unnorm_X(Xn: torch.Tensor) -> torch.Tensor:
    return Xn * rng_ + lb


#  sampling
def repair_to_discrete(X_cont_np: np.ndarray, dv_ranges: list[list[float]]) -> np.ndarray:
    X_disc = np.empty_like(X_cont_np, dtype=np.double)

    for i in range(X_cont_np.shape[0]):
        for j in range(X_cont_np.shape[1]):
            vals = np.asarray(dv_ranges[j], dtype=np.double)
            X_disc[i, j] = vals[np.argmin(np.abs(vals - X_cont_np[i, j]))]

    return X_disc


def sample_random_discrete(n: int, dv_ranges: list[list[float]], rng: np.random.Generator) -> np.ndarray:
    X = np.empty((n, len(dv_ranges)), dtype=np.double)
    for i in range(n):
        X[i] = [rng.choice(vals) for vals in dv_ranges]
    return X



# CSV init
def init_csv(log_csv):
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([
                "objectives",
                "is_feasible",
                "variables",
                "constraints",
                "evaluation_time",
                "algorithm_time",
            ])


def append_rows(log_csv, rows):
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for r in rows:
            w.writerow([
                r["objs_original"],
                r["is_feasible"],
                r["x"],
                r["con_raw"],
                r["eval_time"],
                r.get("alg_time", None),
            ])



# Ref point
def make_ref_point(Y: torch.Tensor) -> torch.Tensor:
    y_min = Y.min(dim=0).values
    margin = torch.tensor([0.1, 0.1], dtype=Y.dtype, device=Y.device)
    return y_min - margin



# Feasibility-weighted MESMO
class FeasibilityWeightedMESMO(AcquisitionFunction):
    def __init__(self, mesmo_acq, constraint_model, threshold: float = 0.0):
        super().__init__(model=mesmo_acq.model)
        self.mesmo_acq = mesmo_acq
        self.cmodel = constraint_model
        self.threshold = float(threshold)
        self.std_normal = Normal(0.0, 1.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # MESMO value for the whole q-batch
        v = self.mesmo_acq(X)  # shape: batch

        # Constraint GP on g(x) = -cv, feasible iff g(x) >= 0
        post = self.cmodel.posterior(X)
        mean = post.mean.squeeze(-1)                      # (batch, q)
        var = post.variance.squeeze(-1).clamp_min(1e-12) # (batch, q)
        std = var.sqrt()

        z = (mean - self.threshold) / std
        p_point = self.std_normal.cdf(z).clamp(1e-12, 1.0)  # (batch, q)

        # batch feasibility probability
        p_batch = p_point.prod(dim=-1)  # (batch,)

        return v * p_batch


# Main BO loop
if __name__ == "__main__":
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    init_csv(LOG_CSV)

    # initial random design
    t0_init = time.perf_counter()
    X_init = sample_random_discrete(n_init, DV_RANGES, rng)
    init_res = eval_batch(X_init, n_jobs=n_jobs)
    t1_init = time.perf_counter()

    train_X = torch.tensor([r["x"] for r in init_res], dtype=dtype, device=device)
    train_Y = torch.tensor([r["y_max"] for r in init_res], dtype=dtype, device=device)
    train_CV = torch.tensor([[r["cv"]] for r in init_res], dtype=dtype, device=device)

    rows = []
    for k, r in enumerate(init_res):
        r2 = dict(r)
        r2["alg_time"] = float(t1_init - t0_init) if k == 0 else None
        rows.append(r2)
    append_rows(LOG_CSV, rows)

    eval_count = n_init

    while eval_count < num_eval:
        t_alg0 = time.perf_counter()

        Xn = norm_X(train_X)

        # objective GPs
        m1 = SingleTaskGP(Xn, train_Y[:, 0:1].contiguous())
        m2 = SingleTaskGP(Xn, train_Y[:, 1:2].contiguous())
        obj_model = ModelListGP(m1, m2)

        obj_mll = SumMarginalLogLikelihood(obj_model.likelihood, obj_model)
        fit_gpytorch_mll(obj_mll, optimizer_kwargs={"options": {"maxiter": 50}})

        # constraint GP: g(x) = -cv, feasible iff g(x) >= 0
        train_G = (-train_CV).contiguous()
        c_model = SingleTaskGP(Xn, train_G)
        c_mll = ExactMarginalLogLikelihood(c_model.likelihood, c_model)
        fit_gpytorch_mll(c_mll, optimizer_kwargs={"options": {"maxiter": 50}})

        # feasible pool by observed cv
        feas_mask = train_CV.squeeze(-1) <= 1e-12
        Y_pool = train_Y[feas_mask] if feas_mask.any() else train_Y

        nd_mask = is_non_dominated(Y_pool)
        pareto_Y = Y_pool[nd_mask]
        if pareto_Y.numel() == 0:
            pareto_Y = Y_pool

        ref_point = make_ref_point(Y_pool)

        partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        hypercell_bounds = partitioning.get_hypercell_bounds().unsqueeze(0)

        mesmo = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=obj_model,
            hypercell_bounds=hypercell_bounds,
            estimation_type="LB",
            num_samples=mc_samples,
        )

        acq = FeasibilityWeightedMESMO(
            mesmo_acq=mesmo,
            constraint_model=c_model,
            threshold=0.0,
        )

        q = min(n_jobs, num_eval - eval_count)

        bounds = torch.stack([
            torch.zeros(dim, dtype=dtype, device=device),
            torch.ones(dim, dtype=dtype, device=device)
        ], dim=0)

        Xnext_n, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"maxiter": maxiter},
        )

        Xnext_cont = unnorm_X(Xnext_n).detach().cpu().numpy().astype(np.double)
        Xnext = repair_to_discrete(Xnext_cont, DV_RANGES)

        t_alg1 = time.perf_counter()
        alg_time = float(t_alg1 - t_alg0)  # <-- 改成 t_alg1

        res_list = eval_batch(Xnext, n_jobs=q)

        X_new = torch.tensor([r["x"] for r in res_list], dtype=dtype, device=device)
        Y_new = torch.tensor([r["y_max"] for r in res_list], dtype=dtype, device=device)
        CV_new = torch.tensor([[r["cv"]] for r in res_list], dtype=dtype, device=device)

        train_X = torch.cat([train_X, X_new], dim=0)
        train_Y = torch.cat([train_Y, Y_new], dim=0)
        train_CV = torch.cat([train_CV, CV_new], dim=0)

        rows = []
        for r in res_list:
            rr = dict(r)
            rr["alg_time"] = alg_time
            rows.append(rr)

        append_rows(LOG_CSV, rows)

        eval_count += q
        print(f"[Mazda qMESMO-FW] eval_count = {eval_count}/{num_eval}, batch_q={q}")