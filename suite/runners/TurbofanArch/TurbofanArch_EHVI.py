import os
import time
import csv
import warnings

import numpy as np
import pandas as pd
import torch

from joblib import Parallel, delayed

from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.exceptions import InputDataWarning

from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch

warnings.filterwarnings("ignore", category=InputDataWarning)

torch.set_default_dtype(torch.double)

# =========================
# User Settings
# =========================
seed = 332
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Budget + batching
TOTAL_BUDGET = 500
num_random_sample = 50

N_JOBS = 5
BATCH_SIZE = N_JOBS

MC_SAMPLES = 16
GP_MAXITER = 50

# Continuous optimize_acqf settings
NUM_RESTARTS = 5
RAW_SAMPLES = 128
ACQ_MAXITER = 100

# Paths
PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/SBO_qLogNEHVI"
os.makedirs(PATH_RESULT, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"TurbofanArch_qLogNEHVI_seed{seed}.csv")


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
    qLogNEHVI here works in max-space:
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


def make_ref_point(train_obj_max: torch.Tensor):
    """
    Dynamic ref point in max-space.
    Should be dominated by observed points.
    """
    y_min = train_obj_max.min(dim=0).values
    y_max = train_obj_max.max(dim=0).values
    span = (y_max - y_min).clamp_min(1e-8)
    ref = y_min - 0.1 * span - 1e-3
    return ref.to(dtype=torch.double)


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
    # TOTAL_BUDGET = total black-box evaluations
    # invalid points also consume budget
    # -------------------------
    while eval_id < TOTAL_BUDGET:
        t_alg0 = time.perf_counter()

        # Deduplicate only for GP fitting
        train_X_gp, train_obj_gp, train_cv_gp = deduplicate_training_data(train_X, train_obj, train_cv)
        train_X_n = norm_X(train_X_gp)

        cv_for_gp = train_cv_gp.clone()
        if cv_for_gp.shape[0] > 1 and torch.std(cv_for_gp) < 1e-12:
            cv_for_gp = cv_for_gp + 1e-6 * torch.randn_like(cv_for_gp)

        # 3 objectives + 1 cv
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
        m_cv = SingleTaskGP(
            train_X_n,
            cv_for_gp[:, 0:1].contiguous(),
            outcome_transform=Standardize(m=1),
        )

        model = ModelListGP(m_obj1, m_obj2, m_obj3, m_cv)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": GP_MAXITER}},
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        ref_point = make_ref_point(train_obj_gp)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_n,
            sampler=sampler,
            prune_baseline=True,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
            constraints=[lambda Z: Z[..., 3]],   # feasible if cv <= 0
        )

        q_now = min(BATCH_SIZE, TOTAL_BUDGET - eval_id)

        bounds = torch.stack([
            torch.zeros(d, dtype=torch.double),
            torch.ones(d, dtype=torch.double)
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

        # Correct candidates to legal architecture variables
        vars_next_list = [correct_candidate(base_problem, row.tolist()) for row in X_next_cont]
        vars_next_list = unique_keep_order(vars_next_list)

        # Remove already valid-and-added points if possible
        existing_keys = set(
            tuple(round(float(x), 12) for x in row)
            for row in train_X.detach().cpu().numpy().tolist()
        )
        new_vars_filtered = [
            row for row in vars_next_list
            if tuple(round(float(x), 12) for x in row) not in existing_keys
        ]

        # Fill if correction causes duplicates or collisions
        while len(new_vars_filtered) < q_now:
            extra = sample_random_vars(base_problem)
            key = tuple(round(float(x), 12) for x in extra)
            if key not in existing_keys and key not in [
                tuple(round(float(z), 12) for z in r) for r in new_vars_filtered
            ]:
                new_vars_filtered.append(extra)

        vars_next_list = new_vars_filtered[:q_now]

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
            f"[BO-Turbofan] evals={eval_id}/{TOTAL_BUDGET} "
            f"(last batch q={q_now}, alg_time={alg_time:.3f}s, valid_points_in_gp={train_X.shape[0]})"
        )

    print(f"[Done] Total evaluations: {eval_id}")
    print(f"[Done] Log saved: {LOG_CSV}")