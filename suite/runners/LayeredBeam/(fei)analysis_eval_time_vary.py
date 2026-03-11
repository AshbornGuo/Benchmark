
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib as mpl

# # =====================
# # Paths
# # =====================
# BASE_DIR = Path.cwd() / "results" / "LayeredBeam"

# time_df = pd.read_csv(BASE_DIR / "algo_time_all.csv")
# hv_df = pd.read_csv(BASE_DIR / "HV_all.csv")

# algo_names = [c for c in time_df.columns if c != "eval_axis"]

# # =====================
# # Budget grid (paper-style)
# # =====================
# time_budget = np.float_power(2, np.arange(-11, 17, 1.5))
# eval_time = np.float_power(2, np.arange(-13, 17, 1.5))

# # =====================
# # Marker style
# # =====================
# markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']
# marker_map = {a: markers[i % len(markers)] for i, a in enumerate(algo_names)}

# # =====================
# # Winner storage
# # =====================
# xbudget = []
# xeval = []
# winner_algo = []
# winner_hv = []

# # =====================
# # Core computation
# # =====================
# for tb in time_budget:
#     for et in eval_time:

#         if 5 * et >= tb:
#             continue

#         results = []

#         for algo in algo_names:
#             algo_time = time_df[algo].values
#             hv = hv_df[algo].values

#             # total_time = 0.0
#             # idx = 0

#             # while idx < len(hv) and total_time < tb:
#             #     total_time += et + algo_time[idx]
#             #     idx += 1

#             # if idx == 0:
#             #     continue

#             # idx = min(idx - 1, len(hv) - 1)
#             # results.append((algo, hv[idx]))
#             completed_idx = -1
#             total_time = 0.0

#             for idx in range(len(hv)):
#                 step_time = et + algo_time[idx]

#             if total_time + step_time > tb:
#                 break

#             total_time += step_time
#             completed_idx = idx

#             if completed_idx == -1:
#                 continue

#             results.append((algo, hv[completed_idx]))

#         if len(results) == 0:
#             continue

#         best_algo, best_hv = max(results, key=lambda x: x[1])   # HV 越大越好

#         xbudget.append(tb)
#         xeval.append(et)
#         winner_algo.append(best_algo)
#         winner_hv.append(best_hv)

# # =====================
# # Convert to numpy
# # =====================
# xbudget = np.array(xbudget, dtype=float)
# xeval = np.array(xeval, dtype=float)
# winner_hv = np.array(winner_hv, dtype=float)
# winner_algo = np.array(winner_algo)


# fig, ax = plt.subplots(figsize=(11.5, 7.2))

# cmap = cm.viridis
# norm = mpl.colors.Normalize(vmin=winner_hv.min(), vmax=winner_hv.max())

# for algo in np.unique(winner_algo):
#     mask = winner_algo == algo

#     ax.scatter(
#         xbudget[mask],
#         xeval[mask],
#         marker=marker_map[algo],
#         c=winner_hv[mask],      # 直接用原始 HV，和参考代码一致
#         cmap=cmap,
#         norm=norm,
#         s=70,
#         edgecolors='black',
#         linewidths=0.8,
#         label=algo,
#         alpha=1.0
#     )

# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("Time budget (sec)")
# ax.set_ylabel("Evaluation time (sec)")
# ax.set_title("LB problem", pad=12)

# ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
# ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)

# # colorbar：按参考代码风格，直接真实 HV
# sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])
# cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, aspect=30)
# cb.set_label("HV", rotation=90, labelpad=10)

# # 只额外关闭 offset，避免 1e-5 + 1.2099 这种难看的显示
# cb.formatter.set_useOffset(False)
# cb.update_ticks()

# # legend
# leg = ax.legend(
#     title="Algorithm",
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.16),
#     ncol=min(len(algo_names), 4),
#     frameon=True,
#     fancybox=True,
#     columnspacing=1.4,
#     handletextpad=0.6,
#     borderpad=0.6
# )

# for h in leg.legend_handles:
#     try:
#         h.set_facecolor("black")
#         h.set_edgecolor("black")
#     except Exception:
#         pass

# plt.subplots_adjust(
#     left=0.10,
#     right=0.86,
#     top=0.90,
#     bottom=0.25
# )

# plt.show()


###########################
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# =====================
# Paths
# =====================
BASE_DIR = Path.cwd() / "results" / "LayeredBeam"

time_df = pd.read_csv(BASE_DIR / "algo_time_all.csv")
hv_df = pd.read_csv(BASE_DIR / "HV_all.csv")


# get max min for plot colour
data = hv_df.drop(columns=["eval_axis"])

global_max = data.to_numpy().max()
global_min = data.to_numpy().min()

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

eval_time_variation = np.float_power(2, np.arange(-13, 17, 1.5))
time_budget_list = np.float_power(2, np.arange(-11, 17, 1.5))

eval_time = 160
time_budget_test = 19180000
algo_name = 'RandomSearch'

def num(time_df, algo_name, eval_time, time_budget):

    # 先执行 50 次初始评估
    init_eval = 50
    init_time =  (init_eval // 5) * eval_time

    if time_budget <= init_time:
        # 连 50 次都做不完
        return int(time_budget // eval_time * 5)

    # 剩余时间用于优化
    remaining_budget = time_budget - init_time

    total_time = 0
    last_eval = init_eval

    for _, row in time_df.iterrows():

        algo_time = row[algo_name]
        step_time = eval_time + algo_time

        if total_time + step_time > remaining_budget:
            break

        total_time += step_time
        last_eval = int(row["eval_axis"])

    return last_eval

def get_hv(hv_df, algo_name, n_eval):

    if n_eval < 55:
        return 0

    return hv_df[hv_df["eval_axis"] <= n_eval].iloc[-1][algo_name]

results = []


n_eval = num(time_df, algo_name, eval_time, time_budget_test)

hv_value = get_hv(hv_df, algo_name, n_eval)

print("n_eval:", n_eval)
print("HV:", hv_value)


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

results_df.to_csv(BASE_DIR / "eval_time_budget_HV_results.csv", index=False)


####################################################

