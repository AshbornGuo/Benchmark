
from pathlib import Path
import pandas as pd
import numpy as np



BASE_DIR = Path.cwd() / "results" / "TurbofanArch"

time_df = pd.read_csv(BASE_DIR / "algo_time_all.csv")
hv_df = pd.read_csv(BASE_DIR / "HV_all.csv")


data = hv_df.drop(columns=["eval_axis"])

algo_names = data.columns.tolist()


eval_time_variation = np.float_power(2, np.arange(-3, 17.1, 1.0))
time_budget_list = np.float_power(2, np.arange(0, 20.1, 1.0))



def num(time_df, algo_name, eval_time, time_budget):
    init_eval = 50
    init_time = (init_eval // 5) * eval_time

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

    extra_blocks = int((remaining_budget - total_time) // eval_time)
    return last_eval + extra_blocks * 5


def get_hv(hv_df, algo_name, n_eval):
    max_eval = int(hv_df["eval_axis"].max())

    if n_eval < 55:
        return np.nan


    if n_eval > max_eval:
        n_eval = max_eval

    return hv_df.loc[hv_df["eval_axis"] == n_eval, algo_name].iloc[0]

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



