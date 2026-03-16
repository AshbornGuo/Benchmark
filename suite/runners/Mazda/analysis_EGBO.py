

import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

# read csv to get reuslts of algorithms   

PATH_EGBO_331 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed331.csv"
PATH_EGBO_332 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed332.csv"
PATH_EGBO_333 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed333.csv"
# PATH_NSGA2_334 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed334.csv"
# PATH_NSGA2_335 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed335.csv"


def read_res(PATH_RESULT):
    # read the results CSV file
    df = pd.read_csv(PATH_RESULT,sep = ";")

    # 1) 把字符串 "[...]" 安全解析成 Python list
    objs_list = df["objectives_original"].map(ast.literal_eval)
    cons_list = df["is_feasible"]

    # 2) 变成 (N, D) 的 ndarray
    F = np.vstack(objs_list.to_numpy())
    F = F[:, 0: 2]

    # 3) 取前两个目标
    f1 = F[:, 0]
    f2 = F[:, 1]

    # normalized
    f1_nor = f1 - 2.0                     # = f1 - 2.0
    f2_nor = f2 / 74.0                    #  对应 common_parts/74，越大越好

    F_nor = np.column_stack([f1_nor, f2_nor])   # shape: (N, 2)

    return cons_list, F_nor


def hypervolume(F):
    if F is None or len(F) == 0:
        return 0.0
    # nondominate
    front0_idx = NonDominatedSorting().do(F)[0] # index of nondominated solution
    F_nd = F[front0_idx] # nondominated set

    # compute hypervolune
    ref_point = np.array([1.1, 0.0])
    hv = HV(ref_point=ref_point).do(F_nd)

    return hv


def hv_analysis(F, fea_inx, step = 50):
    HV = []
    for i in range(step-1, len(F), step):
        # filter feasible solutions
        cons_feasible = fea_inx[:i+1]
        F_step = F[:i+1]
        F_feasible = F_step[cons_feasible]
 

        hv = hypervolume(F_feasible)
        HV.append(hv)

    return HV


# feasible solution ratio
def feas_ratio(df, step = 50) -> list:
    feasible_ratios = []

    for i in range(step-1, len(df), step):
        feasible_ratio=df.iloc[:i+1].mean()
        feasible_ratios.append(feasible_ratio)

    return feasible_ratios


# 
step = 1

# constaints, normalized objectives 
cons_EGBO_331, Fnor_EGBO_331 = read_res(PATH_EGBO_331)
cons_EGBO_332, Fnor_EGBO_332 = read_res(PATH_EGBO_332)
cons_EGBO_333, Fnor_EGBO_333 = read_res(PATH_EGBO_333)
# cons_NSGA2_334, Fnor_NSGA2_334 = read_res(PATH_NSGA2_334)
# cons_NSGA2_335, Fnor_NSGA2_335 = read_res(PATH_NSGA2_335)

# feasible rate
fea_EGBO_331 = feas_ratio(cons_EGBO_331,step)
fea_EGBO_332 = feas_ratio(cons_EGBO_332,step)
fea_EGBO_333 = feas_ratio(cons_EGBO_333,step)
# fea_NSGA2_334 = feas_ratio(cons_NSGA2_334,step)
# fea_NSGA2_335 = feas_ratio(cons_NSGA2_335,step)

# hypervolume
hv_EGBO_331 = hv_analysis(Fnor_EGBO_331, cons_EGBO_331,step)
hv_EGBO_332 = hv_analysis(Fnor_EGBO_332, cons_EGBO_332,step)
hv_EGBO_333 = hv_analysis(Fnor_EGBO_333, cons_EGBO_333,step)
# hv_NSGA2_334 = hv_analysis(Fnor_NSGA2_334, cons_NSGA2_334,step)
# hv_NSGA2_335 = hv_analysis(Fnor_NSGA2_335, cons_NSGA2_335,step)

# take average
hv_runs = [hv_EGBO_331,hv_EGBO_332,hv_EGBO_333]  # 每个是长度 T 的 list
hv_runs = np.array(hv_runs)   # shape: (5, T)
hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)


fea_runs = [fea_EGBO_331,hv_EGBO_332,fea_EGBO_333]  # 每个是长度 T 的 list
fea_runs = np.array(fea_runs)   # shape: (5, T)
fea_mean = fea_runs.mean(axis=0)
fea_std = fea_runs.std(axis=0)
# print(hv_mean)

# plot



T = len(hv_mean)
x = np.arange(1, T + 1) * step



## Mean Feasible Ratio
plt.figure(figsize=(6, 4))

plt.plot(x, fea_mean, label="Mean Feasible Ratio", color="C1")
plt.fill_between(
    x,
    fea_mean - fea_std,
    fea_mean + fea_std,
    color="C1",
    alpha=0.25,
    label="±1 std"
)

plt.xlabel("Evaluations")
plt.ylabel("Feasible Ratio")
plt.ylim(0, 1.05)
plt.title("Feasible Ratio vs Evaluations")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig("results/mazda/SBO_EGBO/Feasible_ratio_curve.png", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(6, 4))

hv_lower = np.maximum(hv_mean - hv_std, 0.0)
hv_upper = hv_mean + hv_std

plt.plot(x, hv_mean, label="Mean HV", color="C0")

plt.fill_between(
    x,
    hv_lower,
    hv_upper,
    color="C0",
    alpha=0.25,
    label="±1 std"
)

plt.xlabel("Evaluations")
plt.ylabel("Hypervolume")
plt.title("Hypervolume vs Evaluations")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/mazda/SBO_EGBO/HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()




#################
##############################
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

## 这是画多层pareto front的
# def plot_first_k_pareto_fronts(csv_path, k=10, save_path="results/mazda/SBO_EGBO/pareto_front.png"):
#     # 1) 读取 CSV
#     df = pd.read_csv(csv_path, sep=";")

#     # 2) 读取 objectives 列
#     objs_list = df["objectives_original"].map(ast.literal_eval)
#     F = np.vstack(objs_list.to_numpy())   # 原目标值 shape (N, 2)
#     F = F[:, 0: 2]

#     # 3) 读取 feasible mask
#     feasible_mask = df["is_feasible"].astype(bool).to_numpy()
#     F_feasible = F[feasible_mask]

#     if len(F_feasible) == 0:
#         print("No feasible solutions found.")
#         return

#     # 4) 两个目标都是最小化，直接做非支配排序
#     fronts = NonDominatedSorting().do(F_feasible)

#     num_fronts = min(k, len(fronts))

#     plt.figure(figsize=(7, 5))
#     cmap = plt.cm.get_cmap("tab10", num_fronts)

#     for i in range(num_fronts):
#         idx = fronts[i]
#         front_points = F_feasible[idx]

#         f1 = front_points[:, 0]
#         f2 = front_points[:, 1]

#         # 按你的要求画图
#         x = -f2
#         y = f1

#         plt.scatter(
#             x,
#             y,
#             s=25,
#             color=cmap(i),
#             label=f"Front {i+1}"
#         )

#         order = np.argsort(x)
#         plt.plot(
#             x[order],
#             y[order],
#             color=cmap(i),
#             alpha=0.7
#         )

#     plt.xlabel("-Objective 2")
#     plt.ylabel("Objective 1")
#     plt.title(f"First {num_fronts} Pareto Front Layers")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print("Saved:", os.path.abspath(save_path))

# plot_first_k_pareto_fronts(
#     r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed331.csv",
#     k=10,
#     save_path=r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Pareto_front_seed331.png"
# )

#####################
## 这是画pareto front和可行解，不可行解的
def plot_pareto_front(csv_path, save_path="results/mazda/SBO_EGBO/Pareto_front_seed333.png"):

    # 1 读取 CSV
    df = pd.read_csv(csv_path, sep=";")

    objs_list = df["objectives_original"].map(ast.literal_eval)
    F = np.vstack(objs_list.to_numpy())
    F = F[:, 0: 2]

    f1 = F[:, 0]
    f2 = F[:, 1]

    feasible_mask = df["is_feasible"].astype(bool).to_numpy()

    # 2 画所有点
    plt.figure(figsize=(7,5))

    # 不可行解
    infeasible = ~feasible_mask
    plt.scatter(
        -f2[infeasible],
        f1[infeasible],
        color="gray",
        s=20,
        label="Infeasible",
        alpha=0.6
    )

    # 可行解
    plt.scatter(
        -f2[feasible_mask],
        f1[feasible_mask],
        color="blue",
        s=20,
        label="Feasible",
        alpha=0.7
    )

    # 3 对可行解做非支配排序
    F_feasible = F[feasible_mask]

    fronts = NonDominatedSorting().do(F_feasible)

    pareto_idx = fronts[0]

    pareto_points = F_feasible[pareto_idx]

    pf1 = pareto_points[:,0]
    pf2 = pareto_points[:,1]

    x_pf = -pf2
    y_pf = pf1

    # 4 画 Pareto front
    plt.scatter(
        x_pf,
        y_pf,
        color="red",
        s=35,
        label="Pareto Front"
    )

    order = np.argsort(x_pf)

    plt.plot(
        x_pf[order],
        y_pf[order],
        color="red",
        linewidth=2
    )

    plt.xlabel("common parts")
    plt.ylabel("total weight")
    plt.title("Pareto Front (EGBO_Seed332)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", os.path.abspath(save_path))


plot_pareto_front(
    r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed333.csv",
    save_path=r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Pareto_front_seed333.png"
)