
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

RESULT_DIR = PROJECT_ROOT / "results" / "LayeredBeam" / "EGBO"



PATH_EGBO_331 = RESULT_DIR / "LayeredBeam_EGBO_seed331.csv"
PATH_EGBO_332 = RESULT_DIR / "LayeredBeam_EGBO_seed332.csv"
PATH_EGBO_333 = RESULT_DIR / "LayeredBeam_EGBO_seed333.csv"


######归一化和参考点都需要改


def read_res(PATH_RESULT):
    # read the results CSV file
    df = pd.read_csv(PATH_RESULT, sep=";")


    # 1) 把字符串 "[...]" 安全解析成 Python list
    objs_list = df["objectives"].map(ast.literal_eval)

    # 2) 变成 (N, D) 的 ndarray
    F = np.vstack(objs_list.to_numpy())

    # 3) 取前两个目标
    f1 = F[:, 0]
    f2 = F[:, 1]
    

    # normalized 
    f1_max = 4.691520000000004;  f1_min = 2.5315200000000093
    f2_max = 13.638929999999998; f2_min =  -2.499999999999873
   
    f1_nor = (f1 - f1_min) / ( f1_max - f1_min) 
 
    f2_nor = (f2 - f2_min) / (f2_max - f2_min)   
      

    F_nor = np.column_stack([f1_nor, f2_nor])   # shape: (N, 2)
    return F_nor





def hypervolume(F):
    if F is None or len(F) == 0:
        return 0.0
    # nondominate
    front0_idx = NonDominatedSorting().do(F)[0] # index of nondominated solution
    F_nd = F[front0_idx] # nondominated set

    # compute hypervolune
    ref_point = np.array([1.1, 1.1])
    hv = HV(ref_point=ref_point).do(F_nd)

    return hv


#     return HV
def hv_analysis(F, step):
    hv_list = []

    for i in range(step, len(F) + 1, step):
        hv = hypervolume(F[:i])   # ← 核心：前 i 个 evaluation
        hv_list.append(hv)

    return hv_list

# 
step = 1

# constaints, normalized objectives 
Fnor_EGBO_331 = read_res(PATH_EGBO_331)
Fnor_EGBO_332 = read_res(PATH_EGBO_332)
Fnor_EGBO_333 = read_res(PATH_EGBO_333)




# # hypervolume
hv_EGBO_331 = hv_analysis(Fnor_EGBO_331, step)
hv_EGBO_332 = hv_analysis(Fnor_EGBO_332, step)
hv_EGBO_333 = hv_analysis(Fnor_EGBO_333, step)


# take average
# hv_runs = [hv_EGBO_331]  # 每个是长度 T 的 list
hv_runs = [hv_EGBO_331, hv_EGBO_332, hv_EGBO_333]  # 每个是长度 T 的 list
hv_runs = np.array(hv_runs)   # shape: (5, T)
hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)


T = len(hv_mean)
x = np.arange(1, T + 1) * step


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
plt.savefig("results/LayeredBeam/EGBO/HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================
# 追加：用“原始 objectives 数值”画非支配分层 + Pareto front 连线
# ============================

def read_res_raw(PATH_RESULT):
    """
    读取 CSV 并返回原始目标值（非归一化），仅取前两个目标
    return: F_raw shape (N,2)
    """
    df = pd.read_csv(PATH_RESULT, sep=";")
    objs_list = df["objectives"].map(ast.literal_eval)
    F = np.vstack(objs_list.to_numpy())
    F_raw = F[:, :2]
    return F_raw


def get_fronts(F):
    """
    返回非支配排序的所有 fronts（分层结果）
    fronts: list of index arrays, fronts[0] 是最优层
    """
    if F is None or len(F) == 0:
        return []
    fronts = NonDominatedSorting().do(F)  # list of arrays
    return fronts


def plot_nd_layers_with_connected_front0(F, title, save_path, max_fronts=None):
    """
    画所有非支配层（分层区分颜色），并把第0层按 f1 排序后连线
    - F: 原始目标值 (N,2)
    - max_fronts: 限制最多画前几层（None 表示全画）
    """
    fronts = get_fronts(F)
    if len(fronts) == 0:
        return

    n_fronts = len(fronts) if max_fronts is None else min(len(fronts), max_fronts)

    plt.figure(figsize=(6, 4))

    # 用离散 colormap 区分层数（层数多也能区分开）
    cmap = plt.get_cmap("tab20", n_fronts)

    # 逐层画点：每一层不同颜色
    for k in range(n_fronts):
        idx = fronts[k]
        Fk = F[idx]
        plt.scatter(Fk[:, 0], Fk[:, 1], s=18, alpha=0.85, color=cmap(k), label=f"Front {k}")


    # 第0层连线（最优层）
    front0 = fronts[0]
    F0 = F[front0]
    if len(F0) > 0:
        order = np.argsort(F0[:, 0])   # 按 f1 排序
        F0_line = F0[order]
        plt.plot(F0_line[:, 0], F0_line[:, 1], linewidth=1.8, label="Front 0 connected")

    plt.xlabel("f1 (raw)")
    plt.ylabel("f2 (raw)")
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# --- 读取原始目标值（非归一化） ---

Front = 5

Fraw_EGBO_331 = read_res_raw(PATH_EGBO_331)


# --- 分别画每个 seed 的分层 Pareto（第0层连线） ---
plot_nd_layers_with_connected_front0(
    Fraw_EGBO_331,
    title="Pareto_front_EGBO_331",
    save_path="results/LayeredBeam/EGBO/Pareto_front_EGBO_331.png",
    max_fronts=Front  # 想只画前K层就填整数，例如 8
)