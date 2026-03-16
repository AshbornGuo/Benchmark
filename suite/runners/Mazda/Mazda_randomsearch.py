import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path


num_evaluation = 1500
seed = 333
rng = np.random.default_rng(seed)

# Paths 
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx" # constraint file
PATH_EXE   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_DV   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/pop_vars_eval.txt"
PATH_RESULT   = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/random_search"

os.makedirs(PATH_RESULT, exist_ok=True)

def dv_range(df_path) -> list:
    """
    retriet the range of dv from txt file
   
    :param df_path: Description
    :return: Description
    """

    dicision_variable = pd.read_excel(df_path)
    volume_lists = []
    for _, row in dicision_variable.iterrows():
        # retriet dv and its values
        dv = row["Design Variable"]
        volume_str = row["Discrete Volume"]

        # check and skill null rows
        if pd.isna(dv) or pd.isna(volume_str):
            continue
    
        # all possible values of a dv
        values = [float(v.strip()) for v in volume_str.split(",")]
        # values of all 222 dv
        volume_lists.append(values)

    return volume_lists 

# run the Mazda problem 
def run_exe(exe_path, input_txt, output_dir):
    # check and creat a new file if not exist
    os.makedirs(output_dir, exist_ok=True) 
    # copy input txt file into output_dir 
    shutil.copyfile(input_txt, os.path.join(output_dir, "pop_vars_eval.txt"))

    subprocess.run([exe_path, output_dir], check=True)

def algo_eval(path_exe, path_dv, path_result, dv_ranges) -> tuple[list[float], bool, list[float], list[float], float, float]:
    """
    retriet the range of dv from txt file
   
    :param df_path: Description
    :return: Description
    :rtype: list
    """
    
    t_0 = time.perf_counter() # start of a random search
    
    # generate a solution usting random search
    with open(path_dv, "w") as f:

        sampled = []
        for values in dv_ranges:
            sampled.append(rng.choice(values))

        f.write("\t".join(map(str, sampled)) + "\n")
    
    t_1 = time.perf_counter() # end of a random search; start of an evaluation

    # run exe and output to path_result
    run_exe(path_exe, path_dv, path_result)

    t_2 = time.perf_counter() # end of an evaluation

    algo_time = t_1 - t_0 # alforithm running time
    eval_time = t_2 - t_1 #evaluation time

    # read 3 txt file(decision variables, ojbctives and constraints condition)
    file_dv = os.path.join(path_result, "pop_vars_eval.txt")
    file_obj = os.path.join(path_result, "pop_objs_eval.txt")
    file_con = os.path.join(path_result, "pop_cons_eval.txt")

    with open(file_obj, "r") as f:
        objs = [float(x) for x in f.read().split()]

    with open(file_dv, "r") as f:
        vars = [float(x) for x in f.read().split()]

    with open(file_con, "r") as f:
        cons = [float(x) for x in f.read().split()]

    # If all constraint values are ≥ 0, return 1 (feasible); otherwise, return 0 (infeasible)
    feasibility = all(x>=0 for x in cons)
    
    # tuple[list[float], int, list[float], list[float], float, float]
    return objs, feasibility, vars, cons, algo_time, eval_time   




# get value ranges of the decision variables
dv = dv_range(PATH_CON)

df = pd.DataFrame(
    columns=[
        "objectives",
        "is_feasible",
        "variables",
        "constraints",
        "algorithm_time",
        "evaluation_time",
    ]
)

for i in range(num_evaluation):
    objs, feas, vars_, cons, t_algo, t_eval = algo_eval(PATH_EXE, PATH_DV, PATH_RESULT, dv)

    df.loc[len(df)] = {
        "objectives": objs,
        "is_feasible": feas,
        "variables": vars_,
        "constraints": cons,
        "algorithm_time": t_algo,
        "evaluation_time": t_eval,
    }


RESULT_DIR = Path(PATH_RESULT)   
RESULT_DIR.mkdir(parents=True, exist_ok=True)
out_file = RESULT_DIR / f"Mazda_randomsearch_seed{seed}.csv"
df.to_csv(out_file, index=False, sep=";")