import os
import time
import csv
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch



seed = 339
population_size = 50
num_eval = 500

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATH_RESULT = PROJECT_ROOT / "results" / "TurbofanArch" / "NSGA2"
PATH_RESULT.mkdir(parents=True, exist_ok=True)

LOG_CSV = PATH_RESULT / f"TurbofanArch_NSGA2_seed{seed}.csv"


def run_one_sim(sim_id: int, vector):

    problem = RealisticTurbofanArch()

    x = np.array(vector, dtype=float)
    X = x[None, :]   # shape = (1, n_var)

    t0 = time.perf_counter()

    X_corr, is_active = problem.correct_x(X)


    F, G = problem.evaluate(X_corr, return_values_of=["F", "G"])

    t1 = time.perf_counter()

    return X_corr, F, G, (t1 - t0)


class MyCallback(Callback):
    def __init__(self, log_csv_path):
        super().__init__()
        self._t_last = None
        self.data["gen_time"] = []
        self.log_csv_path = log_csv_path

    def notify(self, algorithm):
        now = time.perf_counter()
        if self._t_last is None:
            self.data["gen_time"].append(0.0)
        else:
            self.data["gen_time"].append(now - self._t_last)
        self._t_last = now


class TurbofanProblem(ElementwiseProblem):
    def __init__(self, log_csv_path, **kwargs):
        tmp_problem = RealisticTurbofanArch()
        self.log_csv_path = log_csv_path

        super().__init__(
            n_var=tmp_problem.n_var,
            n_obj=tmp_problem.n_obj,
            n_ieq_constr=tmp_problem.n_constr,   
            xl=np.array(tmp_problem.xl, dtype=float),
            xu=np.array(tmp_problem.xu, dtype=float),
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        sim_id = 255 + (uuid.uuid4().int % 1_000_000)
        x_list = np.array(x, dtype=float).tolist()

        X_corr, F, G, eval_time = run_one_sim(sim_id, x_list)

        F = np.array(F, dtype=float).reshape(-1) 
        G = np.array(G, dtype=float).reshape(-1)
        X_corr_flat = np.array(X_corr, dtype=float).reshape(-1)


        is_valid = np.all(np.isfinite(F)) and np.all(np.isfinite(G))

        if not is_valid:
            print(f"[WARNING] Invalid evaluation at sim_id={sim_id}")
            print("x =", x_list)
            print("X_corr =", X_corr_flat.tolist())
            print("F =", F)
            print("G =", G)

            F = np.full(self.n_obj, 1e30, dtype=float)
            G = np.full(self.n_ieq_constr, 1e30, dtype=float)

        is_feasible = np.all(G <= 0)


        out["F"] = F
        out["G"] = G
        out["evaluation_time"] = float(eval_time)


        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                F.tolist(),
                bool(is_valid),
                bool(is_feasible),
                X_corr_flat.tolist(),
                G.tolist(),
                float(eval_time)
            ])


if __name__ == "__main__":
    columns = [
        "objectives",
        "is_valid",
        "is_feasible",
        "variables",
        "constraints",
        "evaluation_time"
    ]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    runner = JoblibParallelization(n_jobs=5, backend="loky")

    problem = TurbofanProblem(
        log_csv_path=LOG_CSV,
        elementwise_runner=runner
    )

    algorithm = NSGA2(pop_size=population_size)

    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", num_eval),
        seed=seed,
        callback=MyCallback(LOG_CSV),
        verbose=True,
    )


    gen_time = res.algorithm.callback.data["gen_time"]

    pd.DataFrame({"gen_time": gen_time}).to_csv(
        PATH_RESULT / f"TurbofanArch_NSGA2_gentime{seed}.csv",
        index=False,
        sep=";",
    )
