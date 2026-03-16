import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
import traceback
import csv
from pathlib import Path
import uuid
from multiprocessing import Manager

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization


random_seed = 333
population_size = 50
num_eval = 1500

PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx"
PATH_EXE = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"

PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2"
os.makedirs(PATH_RESULT, exist_ok=True)
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_NSGA2_seed{random_seed}.csv")

# 每次仿真独立目录（并行关键）
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "MAZDA"


def dv_range(df_path) -> list:
    dicision_variable = pd.read_excel(df_path)
    volume_lists = []
    for _, row in dicision_variable.iterrows():
        dv = row["Design Variable"]
        volume_str = row["Discrete Volume"]
        if pd.isna(dv) or pd.isna(volume_str):
            continue
        values = [float(v.strip()) for v in volume_str.split(",")]
        volume_lists.append(values)
    return volume_lists


def run_exe(exe_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([exe_path, output_dir], check=True)


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._t_last = None
        self.data["gen_time"] = []

    def notify(self, algorithm):
        now = time.perf_counter()
        if self._t_last is None:
            self.data["gen_time"].append(0.0)
        else:
            self.data["gen_time"].append(now - self._t_last)
        self._t_last = now


class Mazda_mop(ElementwiseProblem):

    def __init__(self, dv, eval_counter, eval_lock, **kwargs):
        self.dv = dv
        self.eval_counter = eval_counter
        self.eval_lock = eval_lock

        n_var = len(dv)
        xl = np.zeros(n_var, dtype=int)
        xu = np.array([len(dv[i]) - 1 for i in range(n_var)], dtype=int)

        self.path_exe = PATH_EXE

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=54,
            xl=xl,
            xu=xu,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # ✅ 主进程一致顺序的评估编号（只锁这一小段，不影响并行仿真）
        with self.eval_lock:
            self.eval_counter.value += 1
            eval_id = int(self.eval_counter.value)

        t2 = None
        t3 = None

        # continuous -> discrete（保持你原来的逻辑）
        x_idx = np.rint(x).astype(int)
        x_idx = np.clip(x_idx, self.xl, self.xu).astype(int)
        real_x = [self.dv[i][x_idx[i]] for i in range(len(x_idx))]

        # 每次评估一个独立 workdir（并行关键）
        sim_id = 255 + (uuid.uuid4().int % 1_000_000)
        workdir = RUN_ROOT / f"sim_{sim_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            # 写入 workdir 里的 pop_vars_eval.txt
            dv_file = workdir / "pop_vars_eval.txt"
            with open(dv_file, "w") as f:
                f.write("\t".join(map(str, real_x)) + "\n")

            t2 = time.perf_counter()

            # exe 输出写到 workdir
            run_exe(self.path_exe, str(workdir))

            t3 = time.perf_counter()

            # 从 workdir 读取结果
            file_obj = workdir / "pop_objs_eval.txt"
            file_con = workdir / "pop_cons_eval.txt"
            file_dv  = workdir / "pop_vars_eval.txt"

            with open(file_obj, "r") as f:
                objs = list(map(float, f.read().split()))
            with open(file_dv, "r") as f:
                vars = list(map(float, f.read().split()))
            with open(file_con, "r") as f:
                cons = list(map(float, f.read().split()))

            f1 = float(objs[0])
            f2 = float(objs[1])

            G = -np.array(cons)

            out["F"] = np.array([f1, f2])
            out["G"] = G

        except Exception as e:
            # 保留你原版写法
            print("\n[Evaluate ERROR]", repr(e))
            traceback.print_exc()

            out["F"] = np.array([1e6, 1e6], dtype=float)
            out["G"] = np.ones(54, dtype=float) * 1e6

            objs = [1e6, 1e6]
            vars = real_x
            cons = [1e6] * 54
            G = out["G"]

        eval_time = (t3 - t2) if (t2 is not None and t3 is not None) else None
        is_feasible = np.all(G <= 0)

        # ✅ 每个 workdir 写自己的 result.csv（避免并行写乱）
        local_csv = workdir / "result.csv"
        with open(local_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            # ✅ 写入 eval_id（用于最终合并排序，和串行一致）
            writer.writerow([eval_id, objs, is_feasible, vars, cons, eval_time])


if __name__ == "__main__":

    # 清理 runs
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    dv = dv_range(PATH_CON)

    # 共享计数器：保证 eval_id 顺序与串行一致
    manager = Manager()
    eval_counter = manager.Value("i", 0)
    eval_lock = manager.Lock()

    # 先写表头（最终会被合并覆盖写回 LOG_CSV）
    columns = ["eval_id", "objectives", "is_feasible", "variables", "constraints", "evaluation_time"]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    # 并行 runner（关键）
    runner = JoblibParallelization(n_jobs=5, backend="loky")

    problem = Mazda_mop(
        dv,
        eval_counter=eval_counter,
        eval_lock=eval_lock,
        elementwise_runner=runner
    )

    algorithm = NSGA2(pop_size=population_size)

    res = minimize(
        problem,
        algorithm,
        termination=('n_eval', num_eval),
        seed=random_seed,
        callback=MyCallback(),
        verbose=True
    )

    # ✅ 主进程合并所有 result.csv，并按 eval_id 排序，生成最终 LOG_CSV
    header = ["eval_id", "objectives", "is_feasible", "variables", "constraints", "evaluation_time"]

    rows = []
    for fp in RUN_ROOT.glob("sim_*/result.csv"):
        try:
            with open(fp, "r", newline="", encoding="utf-8") as fin:
                r = csv.reader(fin, delimiter=";")
                for row in r:
                    if row:
                        rows.append(row)
        except Exception as e:
            print(f"[Merge WARN] skip {fp}: {repr(e)}")

    rows.sort(key=lambda r: int(r[0]))

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout, delimiter=";")
        w.writerow(header)
        w.writerows(rows)

    print(f"[Merge] merged {len(rows)} rows")

    # gen time
    gen_time = res.algorithm.callback.data["gen_time"]
    pd.DataFrame({"gen_time": gen_time}).to_csv(
        os.path.join(PATH_RESULT, f"Mazda_NSGA2_gentime{random_seed}.csv"),
        index=False,
        sep=";"
    )