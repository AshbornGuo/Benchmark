


###########################
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================
# Paths
# =====================
BASE_DIR = Path.cwd() / "results" / "mazda"

time_df = pd.read_csv(BASE_DIR / "algo_time_all.csv")
hv_df = pd.read_csv(BASE_DIR / "HV_all.csv")


# get max min for plot colour
data = hv_df.drop(columns=["eval_axis"])

# get algorithm name
algo_names = data.columns.tolist()



#############
# eval_time_variation = [1e-2, 3e-2, 5e-2, 7e-2, 9e-2,
#                     1e-1, 3e-1, 5e-1, 7e-1, 9e-1, 
#                     1e0, 3e0, 5e0, 7e0, 9e0,
#                     1e1, 3e1, 5e1, 7e1, 9e1,
#                     1e2, 3e2, 5e2, 7e2, 9e2,]

# time_budget_list = [1e-1, 3e-1, 5e-1, 7e-1, 9e-1, 
#                     1e0, 3e0, 5e0, 7e0, 9e0,
#                     1e1, 3e1, 5e1, 7e1, 9e1,
#                     1e2, 3e2, 5e2, 7e2, 9e2,
#                     1e3, 3e3, 5e3, 7e3, 9e3,]
############
# time_budget_list = np.float_power(2, np.arange(-3, 25, 1.5))
# eval_time_variation = np.float_power(2, np.arange(-5, 20, 1.5))

##########

# eval_time_variation = np.float_power(2, np.arange(-13, 17, 1.5))
# time_budget_list = np.float_power(2, np.arange(-11, 17, 1.5))

eval_time_variation = np.float_power(2, np.arange(-3, 17.1, 1.0))
time_budget_list = np.float_power(2, np.arange(0, 20.1, 1.0))

# eval_time = 160
# time_budget_test = 1760
# algo_name = 'RandomSearch'

def num(time_df, algo_name, eval_time, time_budget):
    init_eval = 50
    init_time = (init_eval // 5) * eval_time

    # 连前50次都做不完
    if time_budget <= init_time:
        return int(time_budget // eval_time) * 5

    remaining_budget = time_budget - init_time
    total_time = 0.0
    last_eval = init_eval

    for _, row in time_df.iterrows():
        algo_time = row[algo_name]
        step_time = eval_time + algo_time

        if total_time + step_time > remaining_budget:
            return last_eval

        total_time += step_time
        last_eval = int(row["eval_axis"])

    # 走到这里说明：日志里的所有block都已经跑完了
    # 剩下的预算继续按 eval_time 往后推
    extra_blocks = int((remaining_budget - total_time) // eval_time)
    return last_eval + extra_blocks * 5

# def get_hv(hv_df, algo_name, n_eval):
#     max_eval = int(hv_df["eval_axis"].max())

#     if n_eval < 55:
#         return np.nan

#     if n_eval > max_eval:
#         return np.nan

#     return hv_df.loc[hv_df["eval_axis"] == n_eval, algo_name].iloc[0]

def get_hv(hv_df, algo_name, n_eval):
    max_eval = int(hv_df["eval_axis"].max())

    if n_eval < 55:
        return np.nan

    # 如果超过最大评估次数，就用最后一次的 HV
    if n_eval > max_eval:
        n_eval = max_eval

    return hv_df.loc[hv_df["eval_axis"] == n_eval, algo_name].iloc[0]

results = []


# n_eval = num(time_df, algo_name, eval_time, time_budget_test)

# hv_value = get_hv(hv_df, algo_name, n_eval)

# print("n_eval:", n_eval)
# print("HV:", hv_value)


results = []


for algo in algo_names:
    for budget in time_budget_list:
        for eval_time in eval_time_variation:

            n_eval = num(time_df, algo, eval_time, budget)
            hv = get_hv(hv_df, algo, n_eval)

            results.append({
                "algo": algo,
                "time_budget": budget,
                "eval_time": eval_time,
                "n_eval": n_eval,
                "HV": hv
            })

results_df = pd.DataFrame(results)

results_df.to_csv(BASE_DIR / "eval_time_budget_HV.csv", index=False)



# results_df = results_df[results_df["n_eval"] <= 1500]

# results_df.to_csv(BASE_DIR / "eval_time_budget_HV.csv", index=False)