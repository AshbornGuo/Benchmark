import os
import time
import csv
import warnings

import numpy as np
import torch

from joblib import Parallel, delayed

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
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

from botorch.exceptions import InputDataWarning
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch

warnings.filterwarnings("ignore", category=InputDataWarning)
torch.set_default_dtype(torch.double)

# =========================
# User Settings
# =========================
seed = 331
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Budget + batching
TOTAL_BUDGET = 500
num_random_sample = 50

N_JOBS = 5
BATCH_SIZE = N_JOBS  # keep parallel evaluation batch size

MC_SAMPLES = 16
GP_MAXITER = 50

# Lighter optimize_acqf settings for PoF stability
NUM_RESTARTS = 1
RAW_SAMPLES = 16
ACQ_MAXITER = 30

# Paths
PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/SBO_MESMO"
os.makedirs(PATH_RESULT, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"TurbofanArch_MESMO_seed{seed}.csv")

# =========================
# Logging init
# =========================
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "eval_id",
            "objectives",
            "is_valid",
            "is_feasible",
            "variables",
            "constraints",
            "evaluation_time",
            "algorithm_time",
        ])

# =========================
# Helpers
# =========================
def calc_cv_from_g_raw(g_raw_list):
    """
    RealisticTurbofanArch / pymoo style:
      G <= 0 feasible
      cv = sum(max(0, G))
    """
    g = torch.tensor(g_raw_list, dtype=torch.double)
    return torch.clamp(g, min=0.0).sum().item()


def transform_obj_to_max_space(obj_original):
    """
    RealisticTurbofanArch is a pymoo-style minimization problem.
    MESMO here works in max-space:
      y_obj = -F
    """
    f = np.array(obj_original, dtype=float).reshape(-1)
    return (-f).tolist()


def unique_keep_order(list_of_rows, ndigits=12):
    seen = set()
    out = []
    for row in list_of_rows:
        key = tuple(round(float(x), ndigits) for x in row)
        if key not in seen:
            seen.add(key)
            out.append([float(x) for x in row])
    return out


def sample_random_vars(problem):
    x = problem.xl + (problem.xu - problem.xl) * rng.random(problem.n_var)
    X = x[None, :]
    X_corr, _ = problem.correct_x(X)
    return np.array(X_corr, dtype=float).reshape(-1).tolist()


def correct_candidate(problem, x_row):
    X = np.array(x_row, dtype=float)[None, :]
    X_corr, _ = problem.correct_x(X)
    return np.array(X_corr, dtype=float).reshape(-1).tolist()


def deduplicate_training_data(train_X, train_obj, train_cv, ndigits=12):
    seen = {}
    X_np = train_X.detach().cpu().numpy()

    for i, row in enumerate(X_np):
        key = tuple(round(float(x), ndigits) for x in row)
        if key not in seen:
            seen[key] = i

    idx = list(seen.values())
    idx_t = torch.tensor(idx, dtype=torch.long)

    return train_X[idx_t], train_obj[idx_t], train_cv[idx_t]


def make_ref_point(Y: torch.Tensor) -> torch.Tensor:
    y_min = Y.min(dim=0).values
    margin = torch.tensor([0.1, 0.1, 0.1], dtype=Y.dtype, device=Y.device)
    return y_min - margin


def eval_vars_one(vars_one):
    """
    Returns:
      y_obj        : max-space objectives for BoTorch, or None if invalid
      cv           : scalar constraint violation, or None if invalid
      obj_original : original F from problem.evaluate
      g_raw        : original G from problem.evaluate
      is_valid     : whether F/G are finite
      is_feasible  : whether G <= 0
      eval_time    : evaluation time
      x_corr       : corrected variables
    """
    try:
        problem = RealisticTurbofanArch()

        x = np.array(vars_one, dtype=float)
        X = x[None, :]

        t0 = time.perf_counter()

        X_corr, _ = problem.correct_x(X)
        F, G = problem.evaluate(X_corr, return_values_of=["F", "G"])

        t1 = time.perf_counter()
        eval_time = t1 - t0

        X_corr_flat = np.array(X_corr, dtype=float).reshape(-1).tolist()
        F = np.array(F, dtype=float).reshape(-1)
        G = np.array(G, dtype=float).reshape(-1)

        is_valid = np.all(np.isfinite(F)) and np.all(np.isfinite(G))

        if is_valid:
            y_obj = transform_obj_to_max_space(F)
            cv = calc_cv_from_g_raw(G.tolist())
            is_feasible = bool(np.all(G <= 0))
        else:
            y_obj = None
            cv = None
            is_feasible = False

        return y_obj, cv, F.tolist(), G.tolist(), bool(is_valid), is_feasible, float(eval_time), X_corr_flat

    except Exception as e:
        print("[eval_vars_one ERROR]", repr(e))

        problem = RealisticTurbofanArch()
        obj_original = [np.nan] * problem.n_obj
        g_raw = [np.nan] * problem.n_constr
        y_obj = None
        cv = None
        is_valid = False
        is_feasible = False
        eval_time = None
        x_fallback = np.array(vars_one, dtype=float).reshape(-1).tolist()

        return y_obj, cv, obj_original, g_raw, is_valid, is_feasible, eval_time, x_fallback


# =========================
# PoF-weighted MESMO
# =========================
class FeasibilityWeightedMESMO(AcquisitionFunction):
    def __init__(self, mesmo_acq, constraint_model, threshold: float = 0.0):
        super().__init__(model=mesmo_acq.model)
        self.mesmo_acq = mesmo_acq
        self.cmodel = constraint_model
        self.threshold = float(threshold)
        self.std_normal = Normal(0.0, 1.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # MESMO part
        v = self.mesmo_acq(X)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        # Constraint posterior
        post = self.cmodel.posterior(X)
        mean = post.mean.squeeze(-1)
        var = post.variance.squeeze(-1)

        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
        var = torch.nan_to_num(var, nan=1.0, posinf=1.0, neginf=1.0)

        # PoF protections
        var = var.clamp_min(1e-6)
        std = var.sqrt()

        z = (mean - self.threshold) / std
        z = torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
        z = z.clamp(-5.0, 5.0)

        p_point = self.std_normal.cdf(z)
        p_point = torch.nan_to_num(p_point, nan=0.5, posinf=1.0, neginf=0.0)
        p_point = p_point.clamp(1e-6, 1.0 - 1e-6)

        p_batch = p_point.prod(dim=-1)
        p_batch = torch.nan_to_num(p_batch, nan=1e-6, posinf=1.0, neginf=1e-6)
        p_batch = p_batch.clamp(1e-6, 1.0)

        val = v * p_batch
        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        return val


# =========================
# Main
# =========================
if __name__ == "__main__":
    base_problem = RealisticTurbofanArch()
    d = base_problem.n_var
    m = base_problem.n_obj

    print(f"[Info] #decision vars = {d}")
    print(f"[Info] #objectives = {m}")

    if m != 3:
        raise ValueError(f"Expected 3 objectives for RealisticTurbofanArch, but got {m}.")

    # Bounds for normalization
    lb = torch.tensor(np.array(base_problem.xl, dtype=float), dtype=torch.double)
    ub = torch.tensor(np.array(base_problem.xu, dtype=float), dtype=torch.double)
    rng_ = (ub - lb).clamp_min(1e-12)

    def norm_X(X):
        return (X - lb) / rng_

    def unnorm_X(Xn):
        return Xn * rng_ + lb

    # -------------------------
    # Initial sampling (parallel)
    # -------------------------
    t_RS = time.perf_counter()

    init_vars_list = [sample_random_vars(base_problem) for _ in range(num_random_sample)]
    init_vars_list = unique_keep_order(init_vars_list)
    while len(init_vars_list) < num_random_sample:
        init_vars_list += [sample_random_vars(base_problem) for _ in range(num_random_sample)]
        init_vars_list = unique_keep_order(init_vars_list)
    init_vars_list = init_vars_list[:num_random_sample]

    init_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(eval_vars_one)(v) for v in init_vars_list
    )

    t_RE = time.perf_counter()
    print(f"[Init] done {num_random_sample} evals in {t_RE - t_RS:.3f}s")

    # Build initial train tensors
    train_X_list = []
    train_obj_list = []
    train_cv_list = []

    eval_id = 0
    for vars_one, result in zip(init_vars_list, init_results):
        y_obj, cv, obj_original, g_raw, is_valid, is_feasible, eval_time, x_corr = result

        eval_id += 1

        if is_valid:
            train_X_list.append(x_corr)
            train_obj_list.append(y_obj)
            train_cv_list.append([cv])

        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                eval_id,
                obj_original,
                is_valid,
                is_feasible,
                x_corr,
                g_raw,
                eval_time,
                None,
            ])

    if len(train_X_list) < 5:
        raise RuntimeError(
            f"Only {len(train_X_list)} valid initial samples were obtained; too few for stable GP fitting."
        )

    train_X = torch.tensor(train_X_list, dtype=torch.double)
    train_obj = torch.tensor(train_obj_list, dtype=torch.double)  # (n, 3) max-space
    train_cv = torch.tensor(train_cv_list, dtype=torch.double)    # (n, 1)

    # -------------------------
    # BO loop until TOTAL_BUDGET
    # -------------------------
    while eval_id < TOTAL_BUDGET:
        t_alg0 = time.perf_counter()

        # Deduplicate only for GP fitting
        train_X_gp, train_obj_gp, train_cv_gp = deduplicate_training_data(train_X, train_obj, train_cv)
        train_X_n = norm_X(train_X_gp)

        cv_for_gp = train_cv_gp.clone()
        if cv_for_gp.shape[0] > 1 and torch.std(cv_for_gp) < 1e-12:
            cv_for_gp = cv_for_gp + 1e-6 * torch.randn_like(cv_for_gp)

        # objective GPs
        m_obj1 = SingleTaskGP(
            train_X_n,
            train_obj_gp[:, 0:1].contiguous(),
            outcome_transform=Standardize(m=1),
        )
        m_obj2 = SingleTaskGP(
            train_X_n,
            train_obj_gp[:, 1:2].contiguous(),
            outcome_transform=Standardize(m=1),
        )
        m_obj3 = SingleTaskGP(
            train_X_n,
            train_obj_gp[:, 2:3].contiguous(),
            outcome_transform=Standardize(m=1),
        )

        obj_model = ModelListGP(m_obj1, m_obj2, m_obj3)

        obj_mll = SumMarginalLogLikelihood(obj_model.likelihood, obj_model)
        fit_gpytorch_mll(
            obj_mll,
            optimizer_kwargs={"options": {"maxiter": GP_MAXITER}},
        )

        # constraint GP: g_feas(x) = -cv, feasible iff g_feas(x) >= 0
        train_G = (-cv_for_gp).contiguous()
        c_model = SingleTaskGP(
            train_X_n,
            train_G,
            outcome_transform=Standardize(m=1),
        )

        c_mll = ExactMarginalLogLikelihood(c_model.likelihood, c_model)
        fit_gpytorch_mll(
            c_mll,
            optimizer_kwargs={"options": {"maxiter": GP_MAXITER}},
        )

        # observed feasible pool
        feas_mask = train_cv_gp.squeeze(-1) <= 1e-12
        Y_pool = train_obj_gp[feas_mask] if feas_mask.any() else train_obj_gp

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
            num_samples=MC_SAMPLES,
        )

        acq = FeasibilityWeightedMESMO(
            mesmo_acq=mesmo,
            constraint_model=c_model,
            threshold=0.0,
        )

        q_now = min(BATCH_SIZE, TOTAL_BUDGET - eval_id)

        bounds = torch.stack([
            torch.zeros(d, dtype=torch.double),
            torch.ones(d, dtype=torch.double)
        ])

        existing_keys = set(
            tuple(round(float(x), 12) for x in row)
            for row in train_X.detach().cpu().numpy().tolist()
        )

        vars_next_list = []

        # Sequentially choose q_now points with q=1 acquisition optimization
        for _ in range(q_now):
            X_next_n, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={"maxiter": ACQ_MAXITER},
            )

            x_next_cont = unnorm_X(X_next_n)[0]
            x_next = correct_candidate(base_problem, x_next_cont.tolist())

            key = tuple(round(float(x), 12) for x in x_next)

            if key not in existing_keys and key not in [
                tuple(round(float(z), 12) for z in row) for row in vars_next_list
            ]:
                vars_next_list.append(x_next)
                existing_keys.add(key)

        # Fill collapsed duplicates with random corrected points
        while len(vars_next_list) < q_now:
            extra = sample_random_vars(base_problem)
            key = tuple(round(float(x), 12) for x in extra)
            if key not in existing_keys:
                vars_next_list.append(extra)
                existing_keys.add(key)

        t_alg1 = time.perf_counter()
        alg_time = t_alg1 - t_alg0

        # Parallel black-box eval for the batch
        batch_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_vars_one)(v) for v in vars_next_list
        )

        # Update train sets + log
        for vars_one, result in zip(vars_next_list, batch_results):
            y_obj, cv, obj_original, g_raw, is_valid, is_feasible, eval_time, x_corr = result

            eval_id += 1

            if is_valid:
                train_X = torch.cat([train_X, torch.tensor([x_corr], dtype=torch.double)], dim=0)
                train_obj = torch.cat([train_obj, torch.tensor([y_obj], dtype=torch.double)], dim=0)
                train_cv = torch.cat([train_cv, torch.tensor([[cv]], dtype=torch.double)], dim=0)

            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    eval_id,
                    obj_original,
                    is_valid,
                    is_feasible,
                    x_corr,
                    g_raw,
                    eval_time,
                    alg_time,
                ])

        print(
            f"[BO-Turbofan-MESMO] evals={eval_id}/{TOTAL_BUDGET} "
            f"(last batch q={q_now}, alg_time={alg_time:.3f}s, valid_points_in_gp={train_X.shape[0]})"
        )

    print(f"[Done] Total evaluations: {eval_id}")
    print(f"[Done] Log saved: {LOG_CSV}")