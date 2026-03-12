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

from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination


# =========================================================
# 0) Paths / imports
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # python3_11_test/
MECHBENCH_ROOT = PROJECT_ROOT / "problem_sets" / "MECHBench"
sys.path.insert(0, str(MECHBENCH_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "LB"

from src import sob


# =========================================================
# 1) Global config
# =========================================================
seed = 333
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

dim = 10
low, high = -5.0, 5.0

num_eval = 500
n_init = 50
n_jobs = 5                 # parallel sims per iteration
batch_size = n_jobs        # q in EGBO / qLogNEHVI
mc_samples = 16
dtype = torch.double
device = torch.device("cpu")

# BoTorch optimize settings
num_restarts = 5
raw_samples = 128
maxiter = 100
gp_maxiter = 50

# EGBO / NSGA settings
nsga_pop_size = 256

PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/LayeredBeam/EGBO"
os.makedirs(PATH_RESULT, exist_ok=True)

LOG_CSV = os.path.join(PATH_RESULT, f"LayeredBeam_EGBO_seed{seed}.csv")


# =========================================================
# 2) Runner options
# =========================================================
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


# =========================================================
# 3) Black-box evaluation
# =========================================================
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

    return out, (t1 - t0), str(workdir)


def eval_one_x(x_np: np.ndarray):
    """
    Evaluate one design x and return:
      - original objectives
      - max-space objectives for BO
      - aggregated constraint violation cv = max(0, intrusion - 50)
    """
    sim_id = 255 + (uuid.uuid4().int % 1_000_000)
    x_list = x_np.tolist()

    objs, eval_time, workdir = run_one_sim(sim_id, x_list)

    mass = float(objs[0])
    intrusion = float(objs[1])

    # max-space objectives for BO
    y = np.array([-mass, -intrusion], dtype=np.double)

    # aggregated constraint violation: feasible iff cv <= 0
    cv = max(0.0, intrusion - 50.0)
    is_feasible = (cv <= 0.0)

    return {
        "sim_id": sim_id,
        "x": x_list,
        "objs_original": [mass, intrusion],
        "y_max": y.tolist(),
        "cv": float(cv),
        "is_feasible": bool(is_feasible),
        "eval_time": float(eval_time),
        "workdir": workdir,
    }


def eval_batch(X_np: np.ndarray, n_jobs: int):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(eval_one_x)(X_np[i]) for i in range(X_np.shape[0])
    )
    return results


# =========================================================
# 4) Normalization helpers
# =========================================================
LOW = torch.full((dim,), low, dtype=dtype, device=device)
HIGH = torch.full((dim,), high, dtype=dtype, device=device)
RNG = (HIGH - LOW).clamp_min(1e-12)

def norm_X(X: torch.Tensor) -> torch.Tensor:
    return (X - LOW) / RNG

def unnorm_X(Xn: torch.Tensor) -> torch.Tensor:
    return Xn * RNG + LOW


# =========================================================
# 5) CSV helpers
# =========================================================
def init_csv(log_csv):
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([
                "eval_id",
                "objectives",
                "objectives_maxspace",
                "cv",
                "is_feasible",
                "variables",
                "evaluation_time",
                "algorithm_time",
                "workdir",
            ])


def append_rows(log_csv, rows):
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for r in rows:
            w.writerow([
                r["eval_id"],
                r["objs_original"],
                r["y_max"],
                r["cv"],
                r["is_feasible"],
                r["x"],
                r["eval_time"],
                r.get("alg_time", None),
                r.get("workdir", None),
            ])


# =========================================================
# 6) Model helpers
# =========================================================
def make_ref_point(Y_obj: torch.Tensor) -> torch.Tensor:
    """
    Y_obj: (n, 2) in max-space.
    ref_point should be dominated by good points.
    """
    y_min = Y_obj.min(dim=0).values
    margin = torch.tensor([0.5, 0.5], dtype=Y_obj.dtype, device=Y_obj.device)
    return y_min - margin


def build_model(train_X_n: torch.Tensor, train_obj: torch.Tensor, train_cv: torch.Tensor):
    """
    2 objectives + 1 aggregated constraint violation
    """
    m_obj1 = SingleTaskGP(train_X_n, train_obj[:, 0:1].contiguous())
    m_obj2 = SingleTaskGP(train_X_n, train_obj[:, 1:2].contiguous())
    m_cv = SingleTaskGP(train_X_n, train_cv[:, 0:1].contiguous())

    model = ModelListGP(m_obj1, m_obj2, m_cv)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def score_candidates_with_acq(acq, candidates_n: torch.Tensor):
    scores = []
    with torch.no_grad():
        for i in range(candidates_n.shape[0]):
            val = acq(candidates_n[i].unsqueeze(0))
            scores.append(val.item())
    return torch.tensor(scores, dtype=candidates_n.dtype, device=candidates_n.device)


# =========================================================
# 7) EGBO candidate generation
# =========================================================
def egbo_generate_batch(
    acq,
    train_X_n: torch.Tensor,
    train_obj: torch.Tensor,
    train_cv: torch.Tensor,
    q: int,
    d: int,
):
    """
    EGBO structure:
      1) qLogNEHVI batch via optimize_acqf
      2) NSGA-III candidate pool via ask/tell/ask
      3) merge candidates
      4) rerank all by acquisition
      5) take top-q
    """
    bounds = torch.stack([
        torch.zeros(d, dtype=train_X_n.dtype, device=train_X_n.device),
        torch.ones(d, dtype=train_X_n.dtype, device=train_X_n.device),
    ])

    # --------------------------
    # 1) BO candidate batch
    # --------------------------
    qnehvi_x, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": maxiter},
    )

    # --------------------------
    # 2) EA candidate pool
    # --------------------------
    pareto_mask = is_non_dominated(train_obj)
    pareto_x = train_X_n[pareto_mask]
    pareto_y = -train_obj[pareto_mask]   # pymoo minimizes
    pareto_con = train_cv[pareto_mask]

    # fallback
    if pareto_x.shape[0] < 2:
        pareto_x = train_X_n
        pareto_y = -train_obj
        pareto_con = train_cv

    algorithm = UNSGA3(
        pop_size=nsga_pop_size,
        ref_dirs=get_reference_directions(
            "energy",
            train_obj.shape[1],   # 2 objectives
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
        seed=seed,
    )

    pop = algorithm.ask()
    pop.set("F", pareto_y.detach().cpu().numpy())
    pop.set("G", pareto_con.detach().cpu().numpy())
    algorithm.tell(infills=pop)

    newpop = algorithm.ask()
    nsga_x = torch.tensor(
        newpop.get("X"),
        dtype=train_X_n.dtype,
        device=train_X_n.device
    )

    # --------------------------
    # 3) Merge candidates
    # --------------------------
    candidates = torch.cat([qnehvi_x, nsga_x], dim=0)

    # --------------------------
    # 4) Acquisition rerank
    # --------------------------
    scores = score_candidates_with_acq(acq, candidates)
    idx = torch.argsort(scores, descending=True)
    best = candidates[idx[:q]]

    return best


# =========================================================
# 8) Main EGBO loop
# =========================================================
if __name__ == "__main__":
    # clean runs
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    init_csv(LOG_CSV)

    # -------------------------
    # 8.1 Initial random design
    # -------------------------
    t0_init = time.perf_counter()
    X_init = rng.uniform(low, high, size=(n_init, dim)).astype(np.double)
    init_res = eval_batch(X_init, n_jobs=n_jobs)
    t1_init = time.perf_counter()

    train_X = torch.tensor([r["x"] for r in init_res], dtype=dtype, device=device)         # (n, d)
    train_obj = torch.tensor([r["y_max"] for r in init_res], dtype=dtype, device=device)   # (n, 2)
    train_cv = torch.tensor([[r["cv"]] for r in init_res], dtype=dtype, device=device)     # (n, 1)

    rows = []
    eval_id = 0
    for k, r in enumerate(init_res):
        eval_id += 1
        rr = dict(r)
        rr["eval_id"] = eval_id
        rr["alg_time"] = float(t1_init - t0_init) if k == 0 else None
        rows.append(rr)
    append_rows(LOG_CSV, rows)

    eval_count = n_init
    round_id = 0

    # -------------------------
    # 8.2 EGBO iterations
    # -------------------------
    while eval_count < num_eval:
        round_id += 1
        t_alg0 = time.perf_counter()

        # normalize X
        train_X_n = norm_X(train_X)

        # fit GP models (2 objectives + 1 CV)
        model, mll = build_model(train_X_n, train_obj, train_cv)
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": gp_maxiter}},
        )

        # ref point: use feasible points if any
        feas_mask = (train_cv.view(-1) <= 0.0)
        Y_feas = train_obj[feas_mask]
        ref_point = make_ref_point(Y_feas if Y_feas.numel() > 0 else train_obj)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_n,
            sampler=sampler,
            prune_baseline=True,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            constraints=[lambda Z: Z[..., 2]],   # cv <= 0 feasible
        )

        q = min(batch_size, num_eval - eval_count)

        # -------------------------
        # EGBO batch proposal
        # -------------------------
        X_next_n = egbo_generate_batch(
            acq=acq,
            train_X_n=train_X_n,
            train_obj=train_obj,
            train_cv=train_cv,
            q=q,
            d=dim,
        )

        X_next = unnorm_X(X_next_n).detach().cpu().numpy().astype(np.double)

        t_alg1 = time.perf_counter()
        alg_time = float(t_alg1 - t_alg0)

        # parallel evaluate q candidates
        res_list = eval_batch(X_next, n_jobs=q)

        # update dataset
        X_new = torch.tensor([r["x"] for r in res_list], dtype=dtype, device=device)
        Y_new = torch.tensor([r["y_max"] for r in res_list], dtype=dtype, device=device)
        CV_new = torch.tensor([[r["cv"]] for r in res_list], dtype=dtype, device=device)

        train_X = torch.cat([train_X, X_new], dim=0)
        train_obj = torch.cat([train_obj, Y_new], dim=0)
        train_cv = torch.cat([train_cv, CV_new], dim=0)

        # log
        rows = []
        for r in res_list:
            eval_id += 1
            rr = dict(r)
            rr["eval_id"] = eval_id
            rr["alg_time"] = alg_time
            rows.append(rr)

        append_rows(LOG_CSV, rows)

        eval_count += q
        n_feas = int((train_cv.view(-1) <= 0).sum().item())

        print(
            f"[EGBO] round={round_id}  "
            f"eval_count={eval_count}/{num_eval}  "
            f"q={q}  "
            f"feasible={n_feas}  "
            f"alg_time={alg_time:.3f}s  "
            f"ref_point={ref_point.tolist()}"
        )

    print(f"[Done] Total evaluations: {eval_count}")
    print(f"[Done] Log saved: {LOG_CSV}")