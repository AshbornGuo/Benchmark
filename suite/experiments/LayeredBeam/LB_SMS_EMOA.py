import os
import sys
import time
import shutil
import platform
from pathlib import Path
import csv
import numpy as np
import pandas as pd


from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.parallelization.joblib import JoblibParallelization
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # python3_11_test/
MECHBENCH_ROOT = PROJECT_ROOT / "problem_sets" / "MECHBench"
sys.path.insert(0, str(MECHBENCH_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "LB"

from src import sob


seed = 339
population_size = 50
num_eval = 500

dim = 10
low, high = -5, 5


RESULT_ROOT = PROJECT_ROOT / "results" / "LayeredBeam" / "SMSEMOA"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_CSV = RESULT_ROOT / f"LayeredBeam_SMSEMOA_seed{seed}.csv"

# build_runner_options
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

def run_one_sim(sim_id: int, vector):
    runnerOptions = build_runner_options()
    metrics = ["mass", "intrusion"]

    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        f = sob.get_problem(2, 10, runnerOptions, metrics, sequential_id_numbering=False)

        t0 = time.perf_counter()
        out = f(vector, sim_id) 
        t1 = time.perf_counter()
    finally:
        os.chdir(old_cwd)

    return out, (t1 - t0)

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


class LayeredBeamProblem(ElementwiseProblem):
    def __init__(self, log_csv_path, **kwargs):
        super().__init__(
            n_var=dim,
            n_obj=2,
            n_constr=0,
            # n_constr=1,
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

        
        out["F"] = np.array([mass, intrusion], dtype=float)
        # out["G"] = np.array([intrusion - 50.0], dtype=float)

        
        out["X"] = x_list
        out["evaluation_time"] = float(eval_time)

        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            intrusion = float(objs[1])
            # is_feasible = intrusion <= 50

            writer.writerow([objs,  x_list, eval_time])
            # writer.writerow([objs, is_feasible, x_list, eval_time])

    
if __name__ == "__main__":
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    columns = ["objectives",  "variables", "evaluation_time"]
    # columns = ["objectives",  "is_feasible", "variables", "evaluation_time"]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    runner = JoblibParallelization(n_jobs=5, backend="loky")

    problem = LayeredBeamProblem(
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


    gen_time = res.algorithm.callback.data["gen_time"]
    pd.DataFrame({"gen_time": gen_time}).to_csv(
        RESULT_ROOT / f"LayeredBeam_SMSEMOA_gentime{seed}.csv",
        index=False,
        sep=";",
    )