import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH_RESULT = PROJECT_ROOT / "results" / "TurbofanArch" / "random_search"


seed = 339
N = 500
MAX_WORKERS = 5


def unique_keep_order(list_of_rows, ndigits=12):
    seen = set()
    out = []
    for row in list_of_rows:
        key = tuple(round(float(x), ndigits) for x in row)
        if key not in seen:
            seen.add(key)
            out.append([float(x) for x in row])
    return out


def sample_random_corrected(problem, rng):

    x = problem.xl + (problem.xu - problem.xl) * rng.random(problem.n_var)
    X = np.array(x, dtype=float)[None, :]
    X_corr, _ = problem.correct_x(X)
    return np.array(X_corr, dtype=float).reshape(-1).tolist()


def evaluate_one(i, x_corr):
    problem = RealisticTurbofanArch()

    X_corr = np.array(x_corr, dtype=float)[None, :]

    t0 = time.perf_counter()
    F, G = problem.evaluate(X_corr, return_values_of=["F", "G"])
    t1 = time.perf_counter()

    F = np.array(F, dtype=float).reshape(-1)
    G = np.array(G, dtype=float).reshape(-1)

    is_valid = np.all(np.isfinite(F)) and np.all(np.isfinite(G))
    is_feasible = bool(is_valid and np.all(G <= 0))

    return {
        "index": i,
        "objectives": F.tolist(),
        "is_valid": bool(is_valid),
        "is_feasible": bool(is_feasible),
        "variables": np.array(X_corr, dtype=float).reshape(-1).tolist(),
        "constraints": G.tolist(),
        "evaluation_time": t1 - t0,
    }


if __name__ == "__main__":
    mp.freeze_support()

    RESULT_DIR = PATH_RESULT
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    out_file = RESULT_DIR / f"TurbofanArch_randomsearch_seed{seed}.csv"
    alg_file = RESULT_DIR / f"TurbofanArch_randomsearch_algtime_seed{seed}.csv"

    base_problem = RealisticTurbofanArch()
    rng = np.random.default_rng(seed)

    t0_rs = time.perf_counter()

    all_samples = []
    while len(all_samples) < N:
        batch = [sample_random_corrected(base_problem, rng) for _ in range(N)]
        all_samples.extend(batch)
        all_samples = unique_keep_order(all_samples)

    all_samples = all_samples[:N]

    t1_rs = time.perf_counter()
    rs_algorithm_time = t1_rs - t0_rs


    pd.DataFrame([
        {
            "evaluations": N,
            "algorithm_time": rs_algorithm_time
        }
    ]).to_csv(alg_file, index=False)


    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(evaluate_one, i, all_samples[i]) for i in range(N)]

        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda d: d["index"])

    df = pd.DataFrame(results)
    df.drop(columns=["index"], inplace=True)
    df.to_csv(out_file, index=False, sep=";")

