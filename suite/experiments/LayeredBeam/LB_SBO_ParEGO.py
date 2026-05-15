import os
import sys
import time
import shutil
import platform
from pathlib import Path
import csv
import uuid
import numpy as np


import torch

from joblib import Parallel, delayed

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.sampling import sample_simplex



 
PROJECT_ROOT = Path(__file__).resolve().parents[3]   
MECHBENCH_ROOT = PROJECT_ROOT / "problem_sets" / "MECHBench"
sys.path.insert(0, str(MECHBENCH_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "LB"

from src import sob



# Global config 

seed = 339
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

dim = 10
low, high = -5.0, 5.0

num_eval = 500           # total black-box eval budget
n_init = 50              # initial random evals
n_jobs = 5               # parallel sims per iteration (q)
mc_samples = 16          # QMC samples for qLogNParEGO 
dtype = torch.double
device = torch.device("cpu")

# BoTorch optimize settings
num_restarts = 5  
raw_samples = 128 
maxiter = 100  


# PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/LayeredBeam/qLogNParEGO"
# os.makedirs(PATH_RESULT, exist_ok=True)

# LOG_CSV = os.path.join(PATH_RESULT, f"LayeredBeam_qLogNParEGO_seed{seed}.csv")


RESULT_ROOT = PROJECT_ROOT / "results" / "LayeredBeam" / "qLogNParEGO"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_CSV = RESULT_ROOT / f"LayeredBeam_qLogNParEGO_seed{seed}.csv"

# Runner options
def build_runner_options():
    linux_system = platform.system() != "Windows"
    if linux_system:
        orss_main_path = "/home/ivanolar/Documents/OpenRadioss2/OpenRadioss_linux64/OpenRadioss/"
    else:
        orss_main_path = r"C:/Users/guoji/Desktop/graduate project/codes/benchmarks/OpenRadioss_win64/OpenRadioss_win64/OpenRadioss"

    return {
        "open_radioss_main_path": orss_main_path,
        "write_vtk": False,
        "np": 1,
        "nt": 1,
        "h_level": 1,
        "gmsh_verbosity": 0,
    }



# Black-box evaluation
def run_one_sim(sim_id: int, vector):
    runnerOptions = build_runner_options()
    metrics = ["mass", "intrusion"]

    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        f = sob.get_problem(2, 10, runnerOptions, metrics, sequential_id_numbering=False)

        t0 = time.perf_counter()
        out = f(vector, sim_id)  
        t1 = time.perf_counter()
    finally:
        os.chdir(old_cwd)

    return out, (t1 - t0)    


def eval_one_x(x_np: np.ndarray):
    """Evaluate a single design x (dim,) and return original objs + max-space y + time."""
    sim_id = 255 + (uuid.uuid4().int % 1_000_000)
    x_list = x_np.tolist()
    objs, eval_time = run_one_sim(sim_id, x_list)

    mass = float(objs[0])
    intrusion = float(objs[1])

    # max-space for qLogNParEGO: maximize
    y = np.array([-mass, -intrusion], dtype=np.double) # NOTE: BO uses scaled objectives (sea/1000) in y_max, but CSV logs original objectives.

    return {
        "sim_id": sim_id,
        "x": x_list,
        "objs_original": [mass, intrusion],
        "y_max": y.tolist(),
        "eval_time": float(eval_time),
    }


def eval_batch(X_np: np.ndarray, n_jobs: int):
    """Parallel evaluate a batch of points X_np (q, dim)."""
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(eval_one_x)(X_np[i]) for i in range(X_np.shape[0])
    )
    return results


# Normalization 
LOW = torch.full((dim,), low, dtype=dtype, device=device)
HIGH = torch.full((dim,), high, dtype=dtype, device=device)
RNG = (HIGH - LOW).clamp_min(1e-12)

def norm_X(X: torch.Tensor) -> torch.Tensor:
    """Map from original space [-5,5] to [0,1]."""
    return (X - LOW) / RNG

def unnorm_X(Xn: torch.Tensor) -> torch.Tensor:
    """Map from [0,1] back to original space [-5,5]."""
    return Xn * RNG + LOW



# CSV init
def init_csv(log_csv):
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([
                "objectives",
                "is_feasible",
                "variables",   
                "evaluation_time",
                "algorithm_time",

            ])


def append_rows(log_csv, rows):
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for r in rows:
            intrusion = r["objs_original"][1]   
            is_feasible = intrusion <= 50       


            w.writerow([
                r["objs_original"],
                is_feasible,
                r["x"],
                r["eval_time"],
                r.get("alg_time", None),
            ])



# Main BO loop with qLogNParEGO
if __name__ == "__main__":
    # clean runs   
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    init_csv(LOG_CSV)

    # initial random design
    t0_init = time.perf_counter()
    X_init = rng.uniform(low, high, size=(n_init, dim)).astype(np.double)
    init_res = eval_batch(X_init, n_jobs=n_jobs)
    t1_init = time.perf_counter()

    # build training tensors
    train_X = torch.tensor([r["x"] for r in init_res], dtype=dtype, device=device)  # (n, d)
    train_Y = torch.tensor([r["y_max"] for r in init_res], dtype=dtype, device=device)  # (n, 2)

    # log initial
    rows = []
    for k, r in enumerate(init_res):
        r2 = dict(r)
        r2["alg_time"] = float(t1_init - t0_init) if k == 0 else None
        rows.append(r2)
    append_rows(LOG_CSV, rows)

    eval_count = n_init

    # BO iterations until budget
    while eval_count < num_eval:
        t_alg0 = time.perf_counter()

        # normalize X for GP
        Xn = norm_X(train_X)

        # build GP models (2 objectives)
        m1 = SingleTaskGP(Xn, train_Y[:, 0:1].contiguous())
        m2 = SingleTaskGP(Xn, train_Y[:, 1:2].contiguous())
        model = ModelListGP(m1, m2)

        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 50}})


        # determine batch size for this round
        q = min(n_jobs, num_eval - eval_count)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))


        acq_list = []
        for _ in range(q):
            w = sample_simplex(2, dtype=dtype, device=device).squeeze(0)  # shape (2,)
            acq_list.append(
                qLogNParEGO(
                    model=model,
                    X_baseline=Xn,
                    scalarization_weights=w,
                    sampler=sampler,
                    prune_baseline=True,
                    objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
                    constraints=[lambda Y: (-50.0 - Y[..., 1])],
                )
            )



        # optimize in normalized space [0,1]^d
        bounds = torch.stack([torch.zeros(dim, dtype=dtype, device=device),
                              torch.ones(dim, dtype=dtype, device=device)], dim=0)
        
        # q=n_jobs
        Xnext_n, _ = optimize_acqf_list(
            acq_function_list=acq_list,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"maxiter": maxiter},
        )

        # back to original space
        Xnext = unnorm_X(Xnext_n).detach().cpu().numpy().astype(np.double)  # (q, d)

        t_alg1 = time.perf_counter()
        alg_time = float(t_alg1 - t_alg0)

        # parallel evaluate q candidates
        res_list = eval_batch(Xnext, n_jobs=q)

        # update dataset
        X_new = torch.tensor([r["x"] for r in res_list], dtype=dtype, device=device)
        Y_new = torch.tensor([r["y_max"] for r in res_list], dtype=dtype, device=device)

        train_X = torch.cat([train_X, X_new], dim=0)
        train_Y = torch.cat([train_Y, Y_new], dim=0)


        rows = []
        for r in res_list:
            rr = dict(r)
            rr["alg_time"] = alg_time
            rows.append(rr)

        append_rows(LOG_CSV, rows)

        eval_count += q
        print(f"[qLogNParEGO] eval_count = {eval_count}/{num_eval}, batch_q={q}")


