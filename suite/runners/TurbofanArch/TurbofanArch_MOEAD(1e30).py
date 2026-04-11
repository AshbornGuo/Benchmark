import os
import time
import shutil
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import uuid

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import ParallelMOEAD

from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch


# =========================
# 全局设置
# =========================
seed = 331
population_size = 45
num_eval = 500
n_jobs = 5

penalty = np.array([2.85, 250.6, 14.5], dtype=float)

# =========================
# 路径
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "Turbofan"

PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/MOEAD"
os.makedirs(PATH_RESULT, exist_ok=True)

LOG_CSV = os.path.join(PATH_RESULT, f"TurbofanArch_MOEAD_seed{seed}.csv")


# =========================
# 单次评估
# =========================
def run_one_sim(sim_id, vector):

    problem = RealisticTurbofanArch()

    x = np.array(vector, dtype=float)
    X = x[None, :]

    t0 = time.perf_counter()

    X_corr, is_active = problem.correct_x(X)
    F, G = problem.evaluate(X_corr, return_values_of=["F", "G"])

    t1 = time.perf_counter()

    return (F, X_corr, G, None), (t1 - t0)


# =========================
# 每代时间记录
# =========================
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


# =========================
# Turbofan 问题定义
# =========================
class TurbofanProblem(ElementwiseProblem):
    def __init__(self, log_csv_path, **kwargs):
        tmp = RealisticTurbofanArch()

        super().__init__(
            n_var=tmp.n_var,
            n_obj=tmp.n_obj,
            n_constr=tmp.n_constr,
            xl=np.array(tmp.xl, dtype=float),
            xu=np.array(tmp.xu, dtype=float),
            **kwargs
        )

        self.log_csv_path = log_csv_path

    def _evaluate(self, x, out, *args, **kwargs):

        sim_id = f"{time.time_ns()}_{uuid.uuid4().hex[:6]}"

        result, eval_time = run_one_sim(sim_id, x)
        F, X_corr, G, _ = result

        F = np.array(F).reshape(-1)
        G = np.array(G).reshape(-1)
        X_corr = np.array(X_corr).reshape(-1)

        # 防止 NaN / inf
        is_valid = np.all(np.isfinite(F)) and np.all(np.isfinite(G))

        if not is_valid:
            F = np.full(self.n_obj, 1e30)
            G = np.full(self.n_constr, 1e30)

        out["F"] = F
        out["G"] = G

        is_feasible = np.all(G <= 0)

        # 保存每个 sim
        workdir = RUN_ROOT / f"sim_{sim_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        local_csv = workdir / "result.csv"
        with open(local_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                F.tolist(),
                bool(is_valid),
                bool(is_feasible),
                X_corr.tolist(),
                G.tolist(),
                float(eval_time)
            ])


# =========================
# 罚函数包装
# =========================
class ConstraintsAsPenaltyMOO(Problem):

    def __init__(self, problem, penalty):
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=0,
            xl=problem.xl,
            xu=problem.xu
        )
        self.problem = problem
        self.penalty = np.array(penalty, dtype=float)

        if self.penalty.shape[0] != problem.n_obj:
            raise ValueError(
                f"penalty length = {self.penalty.shape[0]}, "
                f"but number of objectives = {problem.n_obj}"
            )

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

        out["F"] = F + CV[:, None] * self.penalty[None, :]


# =========================
# 合并日志
# =========================
def merge_sim_logs(run_root, final_csv_path):

    columns = ["objectives", "is_valid", "is_feasible", "variables", "constraints", "evaluation_time"]
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


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    columns = ["objectives", "is_valid", "is_feasible", "variables", "constraints", "evaluation_time"]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    runner = JoblibParallelization(n_jobs=n_jobs, backend="loky")

    problem = TurbofanProblem(
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

    gen_time = res.algorithm.callback.data["gen_time"]
    pd.DataFrame({"gen_time": gen_time}).to_csv(
        os.path.join(PATH_RESULT, f"TurbofanArch_MOEAD_gentime{seed}.csv"),
        index=False,
        sep=";",
    )