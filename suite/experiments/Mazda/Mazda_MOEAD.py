import os
import time
import shutil
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import traceback
import subprocess
import uuid

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import ParallelMOEAD



seed = 339
population_size = 50
num_eval = 1500
n_jobs = 5

penalty = 1e1



# PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx"
# PATH_EXE = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"

# PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD"
# os.makedirs(PATH_RESULT, exist_ok=True)
# LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_MOEAD_seed{seed}.csv")

# SCRIPT_DIR = Path(__file__).resolve().parent
# RUN_ROOT = SCRIPT_DIR / "runs" / "Mazda"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATH_CON = PROJECT_ROOT / "problem_sets" / "mazda_interface" / "Info_test.xlsx"
PATH_EXE = PROJECT_ROOT / "problem_sets" / "mazda_interface" / "mazda_mop.exe"

PATH_RESULT = PROJECT_ROOT / "results" / "mazda" / "MOEAD"
PATH_RESULT.mkdir(parents=True, exist_ok=True)

LOG_CSV = PATH_RESULT / f"Mazda_MOEAD_seed{seed}.csv"

RUN_ROOT = PATH_RESULT / "runs" / "Mazda"


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


dv = dv_range(PATH_CON)
dim = len(dv)


def run_one_sim(sim_id, vector):
    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    file_dv = workdir / "pop_vars_eval.txt"
    file_obj = workdir / "pop_objs_eval.txt"
    file_con = workdir / "pop_cons_eval.txt"

    try:
        with open(file_dv, "w", encoding="utf-8") as f:
            f.write("\t".join(map(str, vector)) + "\n")

        t0 = time.perf_counter()
        subprocess.run([PATH_EXE, str(workdir)], check=True)
        t1 = time.perf_counter()

        with open(file_obj, "r", encoding="utf-8") as f:
            objs = list(map(float, f.read().split()))

        with open(file_dv, "r", encoding="utf-8") as f:
            vars_out = list(map(float, f.read().split()))

        with open(file_con, "r", encoding="utf-8") as f:
            cons = list(map(float, f.read().split()))

    except Exception as e:
        print(f"\n[Evaluate ERROR] sim_id={sim_id}: {repr(e)}")
        traceback.print_exc()

        objs = [1e6, 1e6]
        vars_out = list(vector)
        cons = [-1e6] * 54
        t1 = None
        t0 = None

    eval_time = (t1 - t0) if (t0 is not None and t1 is not None) else np.nan

    return (objs, vars_out, cons, workdir), eval_time


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


class MazdaProblem(ElementwiseProblem):
    def __init__(self, log_csv_path, **kwargs):
        super().__init__(
            n_var=dim,
            n_obj=2,
            n_constr=54,
            xl=np.zeros(dim, dtype=float),
            xu=np.array([len(dv[i]) - 1 for i in range(dim)], dtype=float),
            **kwargs
        )
        self.log_csv_path = log_csv_path
        self.dv = dv

    def _evaluate(self, x, out, *args, **kwargs):
        sim_id = f"{time.time_ns()}_{uuid.uuid4().hex[:6]}"

        x_idx = np.rint(x).astype(int)
        x_idx = np.clip(x_idx, self.xl.astype(int), self.xu.astype(int))
        x_list = [self.dv[i][x_idx[i]] for i in range(len(x_idx))]

        result, eval_time = run_one_sim(sim_id, x_list)
        objs, vars_out, cons, workdir = result

        f1 = float(objs[0])
        f2 = float(objs[1])

        out["F"] = np.array([f1, f2], dtype=float)

        out["G"] = -np.array(cons, dtype=float)

        out["X"] = vars_out
        out["evaluation_time"] = float(eval_time) if not np.isnan(eval_time) else np.nan

        is_feasible = np.all(out["G"] <= 0)

        local_csv = workdir / "result.csv"
        with open(local_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([objs, is_feasible, vars_out, cons, eval_time])


class ConstraintsAsPenaltyMOO(Problem):
    def __init__(self, problem, penalty: float):
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=0,
            xl=problem.xl,
            xu=problem.xu
        )
        self.problem = problem
        self.penalty = float(penalty)

    def _evaluate(self, X, out, *args, **kwargs):
        _out = self.problem.evaluate(
            X,
            return_values_of=["F", "G"],
            return_as_dictionary=True,
            **kwargs
        )

        F = _out["F"]
        G = _out.get("G", None)

        if G is None:
            CV = np.zeros((F.shape[0],), dtype=float)
        else:
            G = np.atleast_2d(G)
            CV = np.sum(np.maximum(G, 0.0), axis=1)

        out["F"] = F + self.penalty * CV[:, None]


def merge_sim_logs(run_root, final_csv_path):
    columns = ["objectives", "is_feasible", "variables", "constraints", "evaluation_time"]
    rows = []

    for sim_dir in sorted(run_root.glob("sim_*")):
        sim_log = sim_dir / "result.csv"
        if sim_log.exists():
            try:
                with open(sim_log, "r", newline="", encoding="utf-8") as fin:
                    reader = csv.reader(fin, delimiter=";")
                    for row in reader:
                        if row:
                            rows.append(row)
            except Exception as e:
                print(f"[Merge WARN] skip {sim_log}: {repr(e)}")

    with open(final_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter=";")
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"[Merge] merged {len(rows)} rows")


if __name__ == "__main__":

    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    columns = ["objectives", "is_feasible", "variables", "constraints", "evaluation_time"]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    runner = JoblibParallelization(n_jobs=n_jobs, backend="loky")

    problem = MazdaProblem(
        log_csv_path=LOG_CSV,
        elementwise_runner=runner
    )

    problem_pen = ConstraintsAsPenaltyMOO(problem, penalty=penalty)


    ref_dirs = get_reference_directions(
        "das-dennis",
        problem.n_obj,
        n_points=population_size,
        seed=seed
    )


    algorithm = ParallelMOEAD(
        ref_dirs=ref_dirs,
        n_offsprings=n_jobs
    )


    res = minimize(
        problem_pen,
        algorithm,
        termination=("n_eval", num_eval),
        seed=seed,
        callback=MyCallback(LOG_CSV),
        verbose=True,
    )

    merge_sim_logs(RUN_ROOT, LOG_CSV)

    # gen_time = res.algorithm.callback.data["gen_time"]
    # pd.DataFrame({"gen_time": gen_time}).to_csv(
    #     os.path.join(PATH_RESULT, f"Mazda_MOEAD_gentime{seed}.csv"),
    #     index=False,
    #     sep=";",
    # )

    gen_time = res.algorithm.callback.data["gen_time"]
    pd.DataFrame({"gen_time": gen_time}).to_csv(
        PATH_RESULT / f"Mazda_MOEAD_gentime{seed}.csv",
        index=False,
        sep=";",
    )