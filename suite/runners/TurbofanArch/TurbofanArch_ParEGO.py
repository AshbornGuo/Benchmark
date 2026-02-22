import os, csv, time
import numpy as np
from pathlib import Path

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from botorch.optim import optimize_acqf_discrete

from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch


# -----------------------
# Config
# -----------------------
seed = 338
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

problem = RealisticTurbofanArch()

BUDGET = 100
R = 50
T = BUDGET - R
K = 128
MC_SAMPLES = 32
BATCH_SIZE = 1

PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/SBO_ParEGO"
Path(PATH_RESULT).mkdir(parents=True, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"TurbofanArch_SBO_ParEGO_seed{seed}.csv")

# Mazda style header
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "objectives",
            "is_feasible",      # 这里按你的口径 = is_valid（非NaN）
            "variables",        # X_corr
            "constraints",      # G
            "evaluation_time",  # correct_x + evaluate
            "algorithm_time",   # propose/select time
        ])


# -----------------------
# Normalize (for GP input)
# -----------------------
xl = torch.tensor(problem.xl, dtype=torch.double)
xu = torch.tensor(problem.xu, dtype=torch.double)
rng_ = (xu - xl).clamp_min(1e-12)

def norm_X(X: torch.Tensor) -> torch.Tensor:
    return (X - xl) / rng_


# -----------------------
# Evaluate one point
# - returns: Xcorr (d,), F_raw (m,), G_raw (c,), is_valid, cv, eval_time
# - NOTE: cv is internal only (not logged)
# -----------------------
def eval_one(x_np: np.ndarray):
    X = x_np[None, :]

    t0 = time.perf_counter()
    X_corr, _ = problem.correct_x(X)
    F, G = problem.evaluate(X_corr, return_values_of=["F", "G"])
    t1 = time.perf_counter()
    eval_time = t1 - t0

    F_raw = F.reshape(-1)
    G_raw = G.reshape(-1)
    Xcorr = X_corr.reshape(-1)

    is_valid = (not np.isnan(F_raw).any()) and (not np.isnan(G_raw).any())

    # internal-only CV for constrained acquisition: cv = sum(max(0, G))
    if is_valid:
        cv = float(np.maximum(G_raw, 0.0).sum())
    else:
        cv = float(1e6)

    return Xcorr, F_raw, G_raw, is_valid, cv, eval_time


# -----------------------
# Data containers for GP training (ONLY valid points)
# BoTorch uses maximize, so we store Y = -F_raw internally (NOT logged)
# -----------------------
train_Xcorr = []
train_Ymax = []
train_CV = []


# -----------------------
# Initial random samples
# -----------------------
for i in range(R):
    # algorithm time: propose x only
    t_alg0 = time.perf_counter()
    x = problem.xl + (problem.xu - problem.xl) * rng.random(problem.n_var)
    t_alg1 = time.perf_counter()
    alg_time = t_alg1 - t_alg0

    Xcorr, F_raw, G_raw, is_valid, cv, eval_time = eval_one(x)

    # log (Mazda style)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([F_raw.tolist(), bool(is_valid), Xcorr.tolist(), G_raw.tolist(), eval_time, alg_time])

    # training: only valid points
    if not is_valid:
        continue

    train_Xcorr.append(Xcorr)
    train_Ymax.append((-F_raw).tolist())   # internal only
    train_CV.append([cv])                  # internal only


# need at least a few valid points to start
if len(train_Xcorr) < 5:
    raise RuntimeError(f"Too few valid points after init: {len(train_Xcorr)}. Increase R or check evaluation.")


train_X = torch.tensor(train_Xcorr, dtype=torch.double)
train_Y = torch.tensor(train_Ymax, dtype=torch.double)   # (n, m) maximize space
train_CV_t = torch.tensor(train_CV, dtype=torch.double)  # (n, 1)
n_obj = train_Y.shape[1]


def build_model(train_X_n, train_Y, train_CV):
    models = [SingleTaskGP(train_X_n, train_Y[:, j:j+1]) for j in range(n_obj)]
    models.append(SingleTaskGP(train_X_n, train_CV))  # internal cv
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll


# -----------------------
# BO loop (ParEGO)
# -----------------------
for it in range(T):
    t_alg0 = time.perf_counter()

    # fit model
    train_X_n = norm_X(train_X)
    model, mll = build_model(train_X_n, train_Y, train_CV_t)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # ParEGO weights + Chebyshev scalarization (on objectives only)
    weights = sample_simplex(n_obj).squeeze(0)
    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_Y)
    parego_objective = GenericMCObjective(lambda Z, X: scalarization(Z[..., :n_obj]))

    # best_f: use feasible points (cv==0) if available
    FEAS_TOL = 1e-8
    is_feas = (train_CV_t.squeeze(-1) <= FEAS_TOL)
    train_Z = torch.cat([train_Y, train_CV_t], dim=-1)  # (n, m+1)
    if is_feas.any():
        best_f = parego_objective(train_Z[is_feas]).max().item()
    else:
        best_f = parego_objective(train_Z).max().item()

    # acquisition with constraint cv<=0 (internal only)
    acq = qLogExpectedImprovement(
        model=model,
        objective=parego_objective,
        best_f=best_f,
        sampler=sampler,
        constraints=[lambda Z: Z[..., -1]],  # last dim = cv
    )

    # candidate set (K), batch repair, then acquisition on X_corr
    Xcand = problem.xl + (problem.xu - problem.xl) * rng.random((K, problem.n_var))
    Xcand_corr, _ = problem.correct_x(Xcand)
    Xcand_corr = Xcand_corr.reshape(K, -1)

    Xcand_corr_t = torch.tensor(Xcand_corr, dtype=torch.double)
    Xcand_corr_n = norm_X(Xcand_corr_t)

    X_next_n, _ = optimize_acqf_discrete(acq_function=acq, choices=Xcand_corr_n, q=BATCH_SIZE)
    idx = torch.cdist(Xcand_corr_n, X_next_n).argmin().item()
    Xcorr_next = Xcand_corr[idx]

    t_alg1 = time.perf_counter()
    alg_time = t_alg1 - t_alg0

    # evaluate selected point
    Xcorr2, F_raw, G_raw, is_valid, cv, eval_time = eval_one(Xcorr_next)

    # log (Mazda style)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([F_raw.tolist(), bool(is_valid), Xcorr2.tolist(), G_raw.tolist(), eval_time, alg_time])

    # update training with valid points only
    if not is_valid:
        print(f"[{it+1}/{T}] invalid eval, skipped. total eval = {R + it + 1}")
        continue

    train_X = torch.cat([train_X, torch.tensor([Xcorr2], dtype=torch.double)], dim=0)
    train_Y = torch.cat([train_Y, torch.tensor([(-F_raw).tolist()], dtype=torch.double)], dim=0)
    train_CV_t = torch.cat([train_CV_t, torch.tensor([[cv]], dtype=torch.double)], dim=0)

    print(f"[{it+1}/{T}] eval = {R + it + 1}, train_n = {train_X.shape[0]}")
