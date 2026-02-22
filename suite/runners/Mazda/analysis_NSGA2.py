

import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

# read csv to get reuslts of algorithms

PATH_NSGA2_331 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed331.csv"
PATH_NSGA2_332 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed332.csv"
PATH_NSGA2_333 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed333.csv"
PATH_NSGA2_334 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed334.csv"
PATH_NSGA2_335 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/NSGA2/Mazda_NSGA2_seed335.csv"


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
cons_NSGA2_331, Fnor_NSGA2_331 = read_res(PATH_NSGA2_331)
cons_NSGA2_332, Fnor_NSGA2_332 = read_res(PATH_NSGA2_332)
cons_NSGA2_333, Fnor_NSGA2_333 = read_res(PATH_NSGA2_333)
cons_NSGA2_334, Fnor_NSGA2_334 = read_res(PATH_NSGA2_334)
cons_NSGA2_335, Fnor_NSGA2_335 = read_res(PATH_NSGA2_335)

# feasible rate
fea_NSGA2_331 = feas_ratio(cons_NSGA2_331,step)
fea_NSGA2_332 = feas_ratio(cons_NSGA2_332,step)
fea_NSGA2_333 = feas_ratio(cons_NSGA2_333,step)
fea_NSGA2_334 = feas_ratio(cons_NSGA2_334,step)
fea_NSGA2_335 = feas_ratio(cons_NSGA2_335,step)

# hypervolume
hv_NSGA2_331 = hv_analysis(Fnor_NSGA2_331, cons_NSGA2_331,step)
hv_NSGA2_332 = hv_analysis(Fnor_NSGA2_332, cons_NSGA2_332,step)
hv_NSGA2_333 = hv_analysis(Fnor_NSGA2_333, cons_NSGA2_333,step)
hv_NSGA2_334 = hv_analysis(Fnor_NSGA2_334, cons_NSGA2_334,step)
hv_NSGA2_335 = hv_analysis(Fnor_NSGA2_335, cons_NSGA2_335,step)

# take average
hv_runs = [hv_NSGA2_331, hv_NSGA2_332, hv_NSGA2_333, hv_NSGA2_334, hv_NSGA2_335]  # 每个是长度 T 的 list
hv_runs = np.array(hv_runs)   # shape: (5, T)
hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)


fea_runs = [fea_NSGA2_331, fea_NSGA2_332, fea_NSGA2_333, fea_NSGA2_334, fea_NSGA2_335]  # 每个是长度 T 的 list
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
plt.savefig("results/mazda/NSGA2/Feasible_ratio_curve.png", dpi=300, bbox_inches="tight")
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
plt.savefig("results/mazda/NSGA2/HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()



################### 打印非支配层级每层有多少个点
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

def count_front_sizes(F, feasible_mask=None, K=800):
    """
    返回 front0..front(K-1) 每层点数。
    - F: (N,2)
    - feasible_mask: 长度 N 的 bool 序列；如果给了就只在可行解上做排序
    """
    F = np.asarray(F)

    if feasible_mask is not None:
        feasible_mask = np.asarray(feasible_mask, dtype=bool)
        idx_all = np.where(feasible_mask)[0]
        F_use = F[idx_all]
    else:
        idx_all = np.arange(len(F))
        F_use = F

    nds = NonDominatedSorting()
    remaining = np.arange(len(F_use))

    sizes = []
    for _ in range(K):
        if remaining.size == 0:
            break
        front_local = nds.do(F_use[remaining])[0]
        sizes.append(len(front_local))

        mask = np.ones(remaining.size, dtype=bool)
        mask[front_local] = False
        remaining = remaining[mask]

    return sizes


K = 800  # 你想看多少层

sizes_331 = count_front_sizes(Fnor_NSGA2_331, feasible_mask=cons_NSGA2_331, K=800)

print("seed331 feasible points =", np.sum(cons_NSGA2_331))
for i, s in enumerate(sizes_331):
    print(f"front {i}: {s} points")



###################画非支配层级和pareto front
################### 画非支配层级 + 最终 Pareto front（front0）
# ===================== 画非支配层级图 + 最终 Pareto front（用原始目标值） =====================
# 目标语义：min f1, max f2
# 做 NDS 时内部转成最小化空间：[f1, -f2]；但画图用原始 (f1, f2)

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def read_raw_objectives(path_csv):
    """
    从 CSV 读取：
    - feasible: bool mask
    - f1, f2: 原始目标值（不归一化）
    """
    df = pd.read_csv(path_csv, sep=";")
    feasible = df["is_feasible"].to_numpy(dtype=bool)

    objs_list = df["objectives"].map(ast.literal_eval)
    F = np.vstack(objs_list.to_numpy())

    f1 = F[:, 0]   # 原始 f1（min）
    f2 = F[:, 1]   # 原始 f2（max）
    return feasible, f1, f2


def get_front_layers_minmax(f1, f2, feasible_mask=None, K=6):
    """
    返回前 K 层 front 的索引（索引相对过滤后的可行点集合）。
    语义：min f1, max f2
    内部用于排序：F_sort = [f1, f2]（全部最小化）
    """
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)

    if feasible_mask is not None:
        feasible_mask = np.asarray(feasible_mask, dtype=bool)
        f1_use = f1[feasible_mask]
        f2_use = f2[feasible_mask]
    else:
        f1_use = f1
        f2_use = f2

    if len(f1_use) == 0:
        return f1_use, f2_use, []

    F_sort = np.column_stack([f1_use, f2_use])  

    nds = NonDominatedSorting()
    remaining = np.arange(len(F_sort))
    layers = []

    for _ in range(K):
        if remaining.size == 0:
            break

        front_local = nds.do(F_sort[remaining])[0]   # 当前剩余点集的 front0（局部索引）
        front_idx = remaining[front_local]           # 转成相对 F_sort 的索引
        layers.append(front_idx)

        # 删除该层
        mask = np.ones(remaining.size, dtype=bool)
        mask[front_local] = False
        remaining = remaining[mask]

    return f1_use, f2_use, layers


def plot_nds_layers_original(path_csv, K=6, title=None, save_path=None,
                             show_all_points=True, connect_pf=True):
    """
    画前 K 层非支配 front（front0~frontK-1）并高亮最终 Pareto front（front0）
    使用原始目标坐标 (f1, f2) 绘图。
    """
    feasible, f1, f2 = read_raw_objectives(path_csv)
    f1_use, f2_use, layers = get_front_layers_minmax(f1, f2, feasible_mask=feasible, K=K) #把重量负数变为正数再画图
    f2_use = -f2_use
    
    if len(f1_use) == 0:
        print("没有可行解（is_feasible 全 False），无法绘图")
        return

    plt.figure(figsize=(6, 4))

    # 背景：所有可行点
    if show_all_points:
        plt.scatter(f2_use, f1_use, s=8, alpha=0.25, label="all feasible")

    # 各层
    for k, idx in enumerate(layers):
        plt.scatter(f2_use[idx], f1_use[idx], s=22, label=f"front {k}")

    # 高亮 Pareto front
    if layers and len(layers[0]) > 0:
        idx0 = layers[0]
        plt.scatter(f2_use[idx0], f1_use[idx0], s=70,
                facecolors="none", edgecolors="k", linewidths=1.2,
                label="Pareto front (0)")

        if connect_pf and len(idx0) >= 2:
            order = np.argsort(f2_use[idx0])          # ⚠ 按横轴排序
            plt.plot(f2_use[idx0][order], f1_use[idx0][order], linewidth=1.2, label="PF line")

    plt.xlabel("f2 (original, max)")
    plt.ylabel("f1 (original, min)")



    plt.title(title if title is not None else f"NDS layers + Pareto front (K={K})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ===================== 用法示例（seed331） =====================
# 把 PATH_NSGA2_331 换成你自己的变量即可
plot_nds_layers_original(
    PATH_NSGA2_331,
    K=6,
    title="NSGA2 seed331: NDS layers + Pareto front (original objectives)",
    save_path="results/mazda/NSGA2/NDS_layers_seed331_original.png"
)