from concurrent.futures import ProcessPoolExecutor, as_completed  # Python 标准并行库
import platform
import time
import sys
from pathlib import Path
import os
import pandas as pd
import csv


# 修改 Python 搜索路径：把 .../problem_sets/MECHBench 加入搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # python3_11_test/
MECHBENCH_ROOT = PROJECT_ROOT / "problem_sets" / "MECHBench"
sys.path.insert(0, str(MECHBENCH_ROOT))

# 全局路径定义
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "runs" / "LB" # 每个仿真单独目录

from src import sob
import random
import multiprocessing as mp

# fix random seed
seed = 333
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

# worker()（核心计算函数）
def worker(sim_id, vector): # 这个函数会在子进程里跑
    runnerOptions = build_runner_options()
    metrics = ["mass", "intrusion"]

    # 创建独立工作目录, 用来存中间结果
    workdir = RUN_ROOT / f"sim_{sim_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    # 切换当前目录
    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)          # 关键：切换目录

        # 获取 MECHBench 问题
        f = sob.get_problem(2, 10, runnerOptions, metrics, sequential_id_numbering=False)
        
        # evaluation_time
        t0 = time.perf_counter()
        out = f(vector, sim_id)
        t1 = time.perf_counter()

    finally:
        os.chdir(old_cwd)          # 务必切回

    return {"sim_id": sim_id, "time": t1 - t0, "out": out}


PATH_RESULT   = r"C:/Users/guoji/Desktop/python3_11_test/results/LayeredBeam/randomsearch"

os.makedirs(PATH_RESULT, exist_ok=True)

LOG_CSV = os.path.join(PATH_RESULT, f"LayeredBeam_RS_seed{seed}.csv")

columns = [
    "objectives",
    "is_feasible",   
    "variables",
    "evaluation_time",
]

# initiate CSV file
df = pd.DataFrame(columns=columns)
df.to_csv(LOG_CSV, index=False, sep=";")

if __name__ == "__main__": # 主入口（只在主进程执行，防止子进程重复执行下面所有代码
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
   
    # generate 1000 candidate solutions randomly
    vectors = [
        [random.uniform(low, high) for _ in range(dim)]
        for _ in range(num*max_workers)
    ]

    t0_all = time.perf_counter()

    # 至少要保证同一次内的5个进程的sim_id必须不同
    all_sim_ids = []

    # 生成 2*5个任务
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

            intrusion = res["out"][1]          # intrusion 是第二个目标
            is_feasible = intrusion <= 50      # 可行性判断


            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([res["out"], is_feasible, vec, res['time'],])




