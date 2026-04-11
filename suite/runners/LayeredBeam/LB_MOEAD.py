import os
import sys
import time
import shutil
import platform
from pathlib import Path
import csv
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import ParallelMOEAD
import uuid


# 路径与导入：把 MECHBench 加进 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # python3_11_test/
MECHBENCH_ROOT = PROJECT_ROOT / "problem_sets" / "MECHBench"
sys.path.insert(0, str(MECHBENCH_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "LB"

from src import sob


# 全局设置：种子、规模、输出 CSV
seed = 333
population_size = 50 # 子问题数量，约等于NSGA2等算法里的种群数量
num_eval = 500
n_jobs = 5

# 罚函数系数：把约束违反度加到每个目标上
penalty = 1e1

dim = 10
low, high = -5, 5

PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/LayeredBeam/MOEAD"
os.makedirs(PATH_RESULT, exist_ok=True)

LOG_CSV = os.path.join(PATH_RESULT, f"LayeredBeam_MOEAD_seed{seed}.csv")


# build_runner_options：区分 Windows/Linux 的 OpenRadioss 路径
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


# run_one_sim：单次仿真的核心执行
def run_one_sim(sim_id: int, vector):
    runnerOptions = build_runner_options()
    metrics = ["mass", "intrusion"]

    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        # 运行仿真
        f = sob.get_problem(2, 10, runnerOptions, metrics, sequential_id_numbering=False)

        t0 = time.perf_counter()
        out = f(vector, sim_id)  # 原始输出：mass, intrusion
        t1 = time.perf_counter()
    finally:
        os.chdir(old_cwd)

    return out, (t1 - t0)


# MyCallback：每一代记录时间
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


# LayeredBeamProblem：定义 pymoo 的“原始约束问题”
class LayeredBeamProblem(ElementwiseProblem):
    def __init__(self, log_csv_path, **kwargs):
        super().__init__(
            n_var=dim,
            n_obj=2,
            n_constr=1,
            xl=np.full(dim, low, dtype=float),
            xu=np.full(dim, high, dtype=float),
            **kwargs
        )
        self.log_csv_path = log_csv_path

    def _evaluate(self, x, out, *args, **kwargs):
        sim_id = 255 + (uuid.uuid4().int % 1_000_000)

        x_list = x.tolist()

        objs, eval_time = run_one_sim(sim_id, x_list)
        mass = float(objs[0])
        intrusion = float(objs[1])

        # 原始目标
        out["F"] = np.array([mass, intrusion], dtype=float)

        # 约束：intrusion <= 50  ->  intrusion - 50 <= 0
        out["G"] = np.array([intrusion - 50.0], dtype=float)

        # 额外信息（不影响算法）
        out["X"] = x_list
        out["evaluation_time"] = float(eval_time)

        # ---- CSV logging（记录原始物理量，不加罚）---
        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            is_feasible = intrusion <= 50.0
            writer.writerow([objs, is_feasible, x_list, eval_time])


class ConstraintsAsPenaltyMOO(Problem):
    """
    把约束 G<=0 转成罚函数，加到每个目标上，并对算法“隐藏约束”（n_constr=0）。
    """
    def __init__(self, problem, penalty: float):
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=0,          # 包装后对算法来说无约束
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

        F = _out["F"]                           # (n, n_obj)
        G = _out.get("G", None)                 # (n, n_constr) or None

        if G is None:
            CV = np.zeros((F.shape[0],), dtype=float)
        else:
            G = np.atleast_2d(G)
            CV = np.sum(np.maximum(G, 0.0), axis=1)   # 违反度：只累计 G>0 的部分

        # 对每个目标都加罚：F_pen = F + penalty * CV
        out["F"] = F + self.penalty * CV[:, None]


if __name__ == "__main__":
    # 清理 runs
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    # 初始化 CSV（记录原始输出）
    columns = ["objectives", "is_feasible", "variables", "evaluation_time"]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    # 并行 runner
    runner = JoblibParallelization(n_jobs=n_jobs, backend="loky")

    # 原始约束问题（内部仍计算 G）
    problem = LayeredBeamProblem(
        log_csv_path=LOG_CSV,
        elementwise_runner=runner
    )

    # 包装成“罚函数无约束问题”
    problem_pen = ConstraintsAsPenaltyMOO(problem, penalty=penalty)

    # MOEA/D 参考方向（2 目标）
    ref_dirs = get_reference_directions(
        "das-dennis",
        problem.n_obj,
        n_points=population_size,
        seed=seed
    )

    # MOEA/D：只用默认参数也行，这里只传 ref_dirs
    algorithm = ParallelMOEAD(ref_dirs=ref_dirs, 
                      n_offsprings=n_jobs) # 每轮提交 5 个新解评估

    res = minimize(
        problem_pen,                 # 注意：用 penalty wrapper
        algorithm,
        termination=("n_eval", num_eval),
        seed=seed,
        callback=MyCallback(LOG_CSV),
        verbose=True,
    )

    # 保存每代耗时
    gen_time = res.algorithm.callback.data["gen_time"]
    pd.DataFrame({"gen_time": gen_time}).to_csv(
        os.path.join(PATH_RESULT, f"LayeredBeam_MOEAD_gentime{seed}.csv"),
        index=False,
        sep=";",
    )