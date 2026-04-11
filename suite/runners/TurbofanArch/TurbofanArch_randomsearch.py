import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch


PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/random_search"
seed = 333
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
    """
    Sample one raw random point, then correct it immediately.
    Return corrected design variables as a flat list.
    """
    x = problem.xl + (problem.xu - problem.xl) * rng.random(problem.n_var)
    X = np.array(x, dtype=float)[None, :]
    X_corr, _ = problem.correct_x(X)
    return np.array(X_corr, dtype=float).reshape(-1).tolist()


def evaluate_one(i, x_corr):
    """
    Evaluate one already-corrected design point.
    """
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

    os.makedirs(PATH_RESULT, exist_ok=True)
    RESULT_DIR = Path(PATH_RESULT)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    out_file = RESULT_DIR / f"TurbofanArch_randomsearch_seed{seed}.csv"
    alg_file = RESULT_DIR / f"TurbofanArch_randomsearch_algtime_seed{seed}.csv"

    # =========================
    # Build corrected-unique random sample set
    # =========================
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

    print(f"RS algorithm time (sampling + correct + dedup): {rs_algorithm_time:.6f} s")
    print(f"Number of corrected-unique samples collected: {len(all_samples)}")

    pd.DataFrame([
        {
            "evaluations": N,
            "algorithm_time": rs_algorithm_time
        }
    ]).to_csv(alg_file, index=False)

    # =========================
    # Parallel evaluation
    # =========================
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(evaluate_one, i, all_samples[i]) for i in range(N)]

        for fut in as_completed(futures):
            results.append(fut.result())

    # Keep original sample order
    results.sort(key=lambda d: d["index"])

    df = pd.DataFrame(results)
    df.drop(columns=["index"], inplace=True)
    df.to_csv(out_file, index=False, sep=";")

    print(f"Results saved to: {out_file}")
    print(f"Algorithm-time file saved to: {alg_file}")