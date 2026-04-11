from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import shutil
import time
import random
import subprocess
from pathlib import Path
import pandas as pd
import csv


# =========================
# Basic settings
# =========================
seed = 333
random.seed(seed)

num_batches = 300          # 批次数
max_workers = 5            # 每批并行评估数
num_evaluation = num_batches * max_workers

PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx"
PATH_EXE = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_RESULT = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/random_search"

RESULT_DIR = Path(PATH_RESULT)
RUN_ROOT = RESULT_DIR / "runs"

os.makedirs(PATH_RESULT, exist_ok=True)


# =========================
# Read design variable ranges
# =========================
def dv_range(df_path) -> list[list[float]]:
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


# =========================
# Worker
# =========================
def worker(task_id, sampled, exe_path, run_root_str):
    run_root = Path(run_root_str)
    workdir = run_root / f"sim_{task_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    file_dv = workdir / "pop_vars_eval.txt"
    file_obj = workdir / "pop_objs_eval.txt"
    file_con = workdir / "pop_cons_eval.txt"

    with open(file_dv, "w", encoding="utf-8") as f:
        f.write("\t".join(map(str, sampled)) + "\n")

    t0 = time.perf_counter()
    subprocess.run([exe_path, str(workdir)], check=True)
    t1 = time.perf_counter()

    with open(file_obj, "r", encoding="utf-8") as f:
        objs = [float(x) for x in f.read().split()]

    with open(file_con, "r", encoding="utf-8") as f:
        cons = [float(x) for x in f.read().split()]

    with open(file_dv, "r", encoding="utf-8") as f:
        vars_ = [float(x) for x in f.read().split()]

    feasibility = all(x >= 0 for x in cons)

    return {
        "task_id": task_id,
        "objectives": objs,
        "is_feasible": feasibility,
        "variables": vars_,
        "constraints": cons,
        "evaluation_time": t1 - t0,
    }


if __name__ == "__main__":
    mp.freeze_support()

    if RUN_ROOT.exists():
        print(f"Cleaning old runs folder: {RUN_ROOT.resolve()}")
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    LOG_CSV = RESULT_DIR / f"Mazda_RS_seed{seed}.csv"
    ALGTIME_CSV = RESULT_DIR / f"Mazda_RS_algtime_seed{seed}.csv"

    columns = [
        # "task_id",
        # "batch_id",
        "objectives",
        "is_feasible",
        "variables",
        "constraints",
        # "algorithm_time",
        "evaluation_time",
    ]
    pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, sep=";")

    dv_ranges = dv_range(PATH_CON)

    # =========================
    # Sampling all candidates
    # =========================
    t0_rs = time.perf_counter()

    all_samples = []
    all_task_ids = []
    all_batch_ids = []

    task_id = 1
    for batch_id in range(1, num_batches + 1):
        for _ in range(max_workers):
            sampled = [random.choice(values) for values in dv_ranges]
            all_samples.append(sampled)
            all_task_ids.append(task_id)
            all_batch_ids.append(batch_id)
            task_id += 1

    t1_rs = time.perf_counter()

    rs_algorithm_time = t1_rs - t0_rs
    algorithm_time_per_batch = rs_algorithm_time / num_batches

    print(f"RS algorithm time (sampling only): {rs_algorithm_time:.6f} s")
    print(f"Algorithm time per batch ({max_workers} evals): {algorithm_time_per_batch:.6f} s")

    # 保存“每批算法时间”
    rows = []
    for batch_id in range(1, num_batches + 1):
        rows.append({
            "evaluations": batch_id * max_workers,
            "algorithm_time": algorithm_time_per_batch
        })
    pd.DataFrame(rows).to_csv(ALGTIME_CSV, index=False)

    # task_id -> batch_id 映射
    task_to_batch = {
        all_task_ids[i]: all_batch_ids[i]
        for i in range(len(all_task_ids))
    }

    # =========================
    # Parallel evaluation
    # =========================
    t0_all = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_info = {}

        for i in range(num_evaluation):
            tid = all_task_ids[i]
            sampled = all_samples[i]

            fut = ex.submit(worker, tid, sampled, PATH_EXE, str(RUN_ROOT))
            fut_to_info[fut] = tid

        for fut in as_completed(fut_to_info):
            tid = fut_to_info[fut]
            batch_id = task_to_batch[tid]

            try:
                res = fut.result()

                with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow([
                        # res["task_id"],
                        # batch_id,
                        res["objectives"],
                        res["is_feasible"],
                        res["variables"],
                        res["constraints"],
                        # algorithm_time_per_batch,
                        res["evaluation_time"],
                    ])

                print(f"Finished task {tid} (batch {batch_id})")

            except Exception as e:
                print(f"Task {tid} failed: {e}")

    t1_all = time.perf_counter()
    print(f"Total wall-clock time: {t1_all - t0_all:.6f} s")
    print(f"Results saved to: {LOG_CSV}")