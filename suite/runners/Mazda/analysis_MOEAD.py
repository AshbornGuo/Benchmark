

import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

# read csv to get reuslts of algorithms  

PATH_MOEAD_331 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_seed331.csv"
PATH_MOEAD_332 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_seed332.csv"
PATH_MOEAD_333 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_seed333.csv"
PATH_MOEAD_334 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_seed334.csv"
PATH_MOEAD_335 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_seed335.csv"


def read_res(PATH_RESULT):
    # read the results CSV file
    df = pd.read_csv(PATH_RESULT,sep = ";")

    # 1) 把字符串 "[...]" 安全解析成 Python list
    objs_list = df["objectives"].map(ast.literal_eval)
    cons_list = df["is_feasible"]

    # 2) 变成 (N, D) 的 ndarray
    F = np.vstack(objs_list.to_numpy())

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
step = 50

# constaints, normalized objectives 
cons_MOEAD_331, Fnor_MOEAD_331 = read_res(PATH_MOEAD_331)
cons_MOEAD_332, Fnor_MOEAD_332 = read_res(PATH_MOEAD_332)
cons_MOEAD_333, Fnor_MOEAD_333 = read_res(PATH_MOEAD_333)
cons_MOEAD_334, Fnor_MOEAD_334 = read_res(PATH_MOEAD_334)
cons_MOEAD_335, Fnor_MOEAD_335 = read_res(PATH_MOEAD_335)

# feasible rate
fea_MOEAD_331 = feas_ratio(cons_MOEAD_331,step)
fea_MOEAD_332 = feas_ratio(cons_MOEAD_332,step)
fea_MOEAD_333 = feas_ratio(cons_MOEAD_333,step)
fea_MOEAD_334 = feas_ratio(cons_MOEAD_334,step)
fea_MOEAD_335 = feas_ratio(cons_MOEAD_335,step)

# hypervolume
hv_MOEAD_331 = hv_analysis(Fnor_MOEAD_331, cons_MOEAD_331,step)
hv_MOEAD_332 = hv_analysis(Fnor_MOEAD_332, cons_MOEAD_332,step)
hv_MOEAD_333 = hv_analysis(Fnor_MOEAD_333, cons_MOEAD_333,step)
hv_MOEAD_334 = hv_analysis(Fnor_MOEAD_334, cons_MOEAD_334,step)
hv_MOEAD_335 = hv_analysis(Fnor_MOEAD_335, cons_MOEAD_335,step)

# take average
hv_runs = [hv_MOEAD_331, hv_MOEAD_332, hv_MOEAD_333, hv_MOEAD_334, hv_MOEAD_335]  # 每个是长度 T 的 list
hv_runs = np.array(hv_runs)   # shape: (5, T)
hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)


fea_runs = [fea_MOEAD_331, fea_MOEAD_332, fea_MOEAD_333, fea_MOEAD_334, fea_MOEAD_335]  # 每个是长度 T 的 list
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
plt.savefig("results/mazda/MOEAD/Feasible_ratio_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# HV
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
plt.savefig("results/mazda/MOEAD/HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()