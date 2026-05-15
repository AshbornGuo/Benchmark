import sys
import time
import shutil
import platform
from pathlib import Path
import csv
import uuid
import numpy as np
import os

import torch

from joblib import Parallel, delayed

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

  
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf

from botorch.acquisition.acquisition import AcquisitionFunction
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch.distributions import Normal

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # python3_11_test/
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
n_init = 50             # initial random evals
n_jobs = 5               # parallel sims per iteration (q)
mc_samples = 16          
dtype = torch.double
device = torch.device("cpu")

# BoTorch optimize settings
num_restarts = 5  
raw_samples = 128 
maxiter = 100  


RESULT_ROOT = PROJECT_ROOT / "results" / "LayeredBeam" / "MESMO"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_CSV = RESULT_ROOT / f"LayeredBeam_MESMO_seed{seed}.csv"

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

    # max-space for MESMO: maximize
    y = np.array([-mass, -intrusion], dtype=np.double) 

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


# Ref point heuristic in max-space  
def make_ref_point(Y: torch.Tensor) -> torch.Tensor:
    """
    Y: (n, 2) in max-space.
    ref_point should be dominated by (worse than) all good points.
    A simple safe choice: min(Y, dim=0) - margin
    """
    y_min = Y.min(dim=0).values
    margin = torch.tensor([0.5, 0.5], dtype=Y.dtype, device=Y.device)
    return (y_min - margin)


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

class FeasibilityWeightedMESMO(AcquisitionFunction):
    def __init__(self, mesmo_acq, constraint_model, threshold: float = 0.0):
        """
        g(x) ~ GP, feasible if g(x) >= threshold.
        acq(x) = MESMO(x) * P(g(x) >= threshold)
        """
        super().__init__(model=mesmo_acq.model)
        self.mesmo_acq = mesmo_acq
        self.cmodel = constraint_model
        self.threshold = float(threshold)
        self.std_normal = Normal(0.0, 1.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        v = self.mesmo_acq(X)  # shape: batch, e.g. (128,)

        post = self.cmodel.posterior(X)
        mean = post.mean.squeeze(-1)                       # (batch, q) e.g. (128,5)
        var = post.variance.squeeze(-1).clamp_min(1e-12)   # (batch, q)
        std = var.sqrt()

        z = (mean - self.threshold) / std
        p_point = self.std_normal.cdf(z).clamp(1e-6, 1.0)  # (batch, q)

    
        p_batch = p_point.prod(dim=-1)

        return v * p_batch
    
if __name__ == "__main__":
    # clean runs   
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    init_csv(LOG_CSV)


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

        # Constraint GP: g(x)=50 - intrusion
        train_C = (50.0 + train_Y[:, 1]).unsqueeze(-1)  # shape (n,1)

        c_model = SingleTaskGP(Xn, train_C.contiguous())
        c_mll = ExactMarginalLogLikelihood(c_model.likelihood, c_model)
        fit_gpytorch_mll(c_mll, optimizer_kwargs={"options": {"maxiter": 50}})

        # intrusion<=50 <=> y2=-intrusion >= -50
        feas_mask = train_Y[:, 1] >= -50.0
        Y_pool = train_Y[feas_mask] if feas_mask.any() else train_Y

        nd_mask = is_non_dominated(Y_pool)
        pareto_Y = Y_pool[nd_mask]
        if pareto_Y.numel() == 0:
            pareto_Y = Y_pool  

        ref_point = make_ref_point(Y_pool)


        partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        hypercell_bounds = partitioning.get_hypercell_bounds().unsqueeze(0)  # -> (1, 2, J, M)


        mesmo = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=model,
            hypercell_bounds=hypercell_bounds,
            estimation_type="LB",
            num_samples=mc_samples,
        )

        acq = FeasibilityWeightedMESMO(mesmo_acq=mesmo, constraint_model=c_model, threshold=0.0)

        

        # determine batch size for this round 
        q = min(n_jobs, num_eval - eval_count)

        # optimize in normalized space [0,1]^d
        bounds = torch.stack([torch.zeros(dim, dtype=dtype, device=device),
                              torch.ones(dim, dtype=dtype, device=device)], dim=0)
        
        # q=n_jobs
        Xnext_n, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
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
        print(f"[MESMO] eval_count = {eval_count}/{num_eval}, batch_q={q}, ref_point={ref_point.tolist()}")





