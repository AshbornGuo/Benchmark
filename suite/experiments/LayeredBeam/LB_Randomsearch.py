from concurrent.futures import ProcessPoolExecutor, as_completed  # Python 标准并行库
import platform
import time
import sys
from pathlib import Path
import os
import pandas as pd
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]   
MECHBENCH_ROOT = PROJECT_ROOT / "problem_sets" / "MECHBench"
sys.path.insert(0, str(MECHBENCH_ROOT))


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "LB" # 每个仿真单独目录

from src import sob
import random
import multiprocessing as mp

# fix random seed
seed = 339
random.seed(seed)  

# build_runner_options()
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


def worker(sim_id, vector): 
    runnerOptions = build_runner_options()
    metrics = ["mass", "intrusion"]

    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)          

        f = sob.get_problem(2, 10, runnerOptions, metrics, sequential_id_numbering=False)
        
        # evaluation_time
        t0 = time.perf_counter()
        out = f(vector, sim_id)
        t1 = time.perf_counter()

    finally:
        os.chdir(old_cwd)         

    return {"sim_id": sim_id, "time": t1 - t0, "out": out}


RESULT_ROOT = PROJECT_ROOT / "results" / "LayeredBeam" / "randomsearch"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_CSV = RESULT_ROOT / f"LayeredBeam_RS_seed{seed}.csv"

columns = [
    "objectives",
    "is_feasible",   
    "variables",
    "evaluation_time",
]

# initiate CSV file
df = pd.DataFrame(columns=columns)
df.to_csv(LOG_CSV, index=False, sep=";")

if __name__ == "__main__": 
    mp.freeze_support()

    import shutil

    # clean historical file before start
    if RUN_ROOT.exists():
        print(f"Cleaning old runs folder: {RUN_ROOT.resolve()}")
        shutil.rmtree(RUN_ROOT)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    
    # num_candidates = num*max_workers
    num = 100
    dim = 10
    low, high = -5, 5
    max_workers = 5
   

    t0_rs = time.perf_counter()
    vectors = [
        [random.uniform(low, high) for _ in range(dim)]
        for _ in range(num * max_workers)
    ]
    t1_rs = time.perf_counter()

    rs_algorithm_time = t1_rs - t0_rs
    print(f"RS algorithm time (sampling only): {rs_algorithm_time:.6f} s")

    rs_time_per_5_evals = rs_algorithm_time / num

    # 写入CSV
    rs_time_csv = RESULT_ROOT / f"LayeredBeam_RS_algtime_seed{seed}.csv"

    rows = []
    for i in range(1, num + 1):
        rows.append({
            # "batch": i,
            "evaluations": i * max_workers,
            "algorithm_time": rs_time_per_5_evals
        })

    df_rs_time = pd.DataFrame(rows)
    df_rs_time.to_csv(rs_time_csv, index=False)


    t0_all = time.perf_counter()

    all_sim_ids = []

    for i in range(num):
        for j in range(max_workers):
            all_sim_ids.append(i * 10 + j + 255)

    results = []

    with ProcessPoolExecutor(max_workers) as ex:
        fut_to_info = {}
        for i in range(len(all_sim_ids)):
            sid = all_sim_ids[i]
            vec = vectors[i]
            fut = ex.submit(worker, sid, vec)
            fut_to_info[fut] = (sid, vec)

        for fut in as_completed(fut_to_info):
            sid, vec = fut_to_info[fut]
            res = fut.result()

            intrusion = res["out"][1]          
            is_feasible = intrusion <= 50     


            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([res["out"], is_feasible, vec, res['time'],])




