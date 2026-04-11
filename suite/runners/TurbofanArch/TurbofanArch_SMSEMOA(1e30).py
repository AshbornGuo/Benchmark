import os
import time
import csv
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization

from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch


# =========================
# 全局设置：种子、规模、输出 CSV
# =========================
seed = 333
population_size = 50
num_eval = 500

PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/SMS_EMOA"
os.makedirs(PATH_RESULT, exist_ok=True)

LOG_CSV = os.path.join(PATH_RESULT, f"TurbofanArch_SMSEMOA_seed{seed}.csv")


# =========================
# 单次评估函数
# 保留“单点评估 + 计时”结构
# =========================
def run_one_sim(sim_id: int, vector):
    # 每次评估单独创建问题对象，避免并行时共享状态
    problem = RealisticTurbofanArch()

    x = np.array(vector, dtype=float)
    X = x[None, :]   # shape = (1, n_var)

    t0 = time.perf_counter()

    # 修复变量
    X_corr, is_active = problem.correct_x(X)

    # 评估目标和约束
    F, G = problem.evaluate(X_corr, return_values_of=["F", "G"])

    t1 = time.perf_counter()

    return X_corr, F, G, (t1 - t0)


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
# 定义 pymoo 问题
# =========================
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

        # 检查是否有效：必须是有限数
        is_valid = np.all(np.isfinite(F)) and np.all(np.isfinite(G))

        # 如果无效，替换成非常差但有限的值，避免算法内部报错
        if not is_valid:
            print(f"[WARNING] Invalid evaluation at sim_id={sim_id}")
            print("x =", x_list)
            print("X_corr =", X_corr_flat.tolist())
            print("F =", F)
            print("G =", G)

            F = np.full(self.n_obj, 1e30, dtype=float)
            G = np.full(self.n_ieq_constr, 1e30, dtype=float)

        is_feasible = np.all(G <= 0)

        # 只给 pymoo 它真正需要的字段
        out["F"] = F
        out["G"] = G
        out["evaluation_time"] = float(eval_time)

        # CSV logging
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


# =========================
# 主程序
# =========================
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

    algorithm = SMSEMOA(pop_size=population_size)

    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", num_eval),
        seed=seed,
        callback=MyCallback(LOG_CSV),
        verbose=True,
    )

    # 保存每代时间
    gen_time = res.algorithm.callback.data["gen_time"]
    pd.DataFrame({"gen_time": gen_time}).to_csv(
        os.path.join(PATH_RESULT, f"TurbofanArch_SMSEMOA_gentime{seed}.csv"),
        index=False,
        sep=";",
    )

    # # 可选：保存最终结果
    # if res.X is not None:
    #     final_df = pd.DataFrame({
    #         "X": [x.tolist() for x in np.atleast_2d(res.X)],
    #         "F": [f.tolist() for f in np.atleast_2d(res.F)]
    #     })
    #     final_df.to_csv(
    #         os.path.join(PATH_RESULT, f"TurbofanArch_SMSEMOA_finalPF_seed{seed}.csv"),
    #         index=False,
    #         sep=";"
    #     )