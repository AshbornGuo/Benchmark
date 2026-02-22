import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
import traceback
import csv
import subprocess
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.callback import Callback


random_seed = 335
population_size = 100
num_eval = 5000

# Paths 
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx" # constraint file
PATH_EXE   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_DV   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/pop_vars_eval.txt"

PATH_RESULT   = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SMS_EMOA"
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_SMS_EMOA_seed{random_seed}.csv")

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

    def __init__(self):
        self.dv = dv
        n_var = len(dv)
        xl = np.zeros(n_var, dtype=int)
        xu = np.array([len(dv[i]) - 1 for i in range(n_var)], dtype=int)
        self.path_exe = PATH_EXE
        self.path_dv = PATH_DV
        self.path_con = PATH_CON
        self.path_result = PATH_RESULT

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=54,
            xl=xl, # lower bound of dv
            xu=xu, # upper bound of dv
        )

    def _evaluate(self, x, out, *args, **kwargs):

        t2 = None
        t3 = None

        # transform a continuouns variable from NSGA-II into a discrete variable
        x_idx = np.rint(x).astype(int)
        x_idx = np.clip(x_idx, self.xl, self.xu).astype(int)
        real_x = [self.dv[i][x_idx[i]] for i in range(len(x_idx))]


        try:
            # write dv into a txt file
            with open(self.path_dv, "w") as f:
                f.write("\t".join(map(str, real_x)) + "\n")
            
            t2 = time.perf_counter()

            # run the problem
            run_exe(self.path_exe, self.path_dv, self.path_result)

            t3 = time.perf_counter()

            # read 3 txt file(decision variables, ojbctives and constraints condition)
            file_dv = os.path.join(self.path_result, "pop_vars_eval.txt")
            file_obj = os.path.join(self.path_result, "pop_objs_eval.txt")
            file_con = os.path.join(self.path_result, "pop_cons_eval.txt")

            # read the result txt file
            with open(file_obj, "r") as f:
                objs = list(map(float, f.read().split()))
            with open(file_dv, "r") as f:
                vars = list(map(float, f.read().split()))
            with open(file_con, "r") as f:
                cons = list(map(float, f.read().split()))

            f1 = float(objs[0])
            f2 = float(objs[1])

            # transforms constraints from >=0 to <= 0 
            G = -np.array(cons)   

            # return to pymoo
            out["F"] = np.array([f1, f2])
            out["G"] = G

        except Exception as e:
            print("\n[Evaluate ERROR]", repr(e))
            traceback.print_exc()

            out["F"] = np.array([1e6, 1e6], dtype=float)
            out["G"] = np.ones(54, dtype=float) * 1e6


        eval_time = (t3 - t2) if (t2 is not None and t3 is not None) else None

        if not hasattr(self, "eval_times"):
            self.eval_times = []
        self.eval_times.append(eval_time)

        is_feasible = np.all(G <= 0)
        # G_out = out["G"]
        # is_feasible = np.all(G_out <= 0)


        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([objs,is_feasible,vars,cons, eval_time,])


# get value ranges of the decision variables
dv = dv_range(PATH_CON)

problem = Mazda_mop()

algorithm = SMSEMOA(
    pop_size=population_size,
)


columns = [
    "objectives",
    "is_feasible",
    "variables",
    "constraints",
    "evaluation_time",
]

df = pd.DataFrame(columns=columns)
df.to_csv(LOG_CSV, index=False, sep=";")

res = minimize(
    problem,
    algorithm,
    termination=('n_eval', num_eval),
    seed=random_seed,
    callback=MyCallback(),
    verbose=True
)

gen_time = res.algorithm.callback.data["gen_time"]
df = pd.DataFrame({"gen_time": gen_time})
df.to_csv(
    os.path.join(PATH_RESULT, f"Mazda_SMS_EMOA_gentime{random_seed}.csv"),
    index=False,
    sep=";"
)




