from pathlib import Path
import pandas as pd
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt


PROJECT_ROOT = Path.cwd()
RESULT_DIR = PROJECT_ROOT / "results" / "TurbofanArch" / "random_search"

PATH_random_search_331 = RESULT_DIR / "TurbofanArch_randomsearch_seed331.csv"
PATH_random_search_332 = RESULT_DIR / "TurbofanArch_randomsearch_seed332.csv"
PATH_random_search_333 = RESULT_DIR / "TurbofanArch_randomsearch_seed333.csv"


# =========================
# 0) 解析 objectives 字符串
#    支持:
#    - "[1.0, 2.0, 3.0]"
#    - "[nan, nan, nan]"
# =========================
def parse_obj_list(s):
    if pd.isna(s):
        return None

    s = str(s).strip()

    try:
        return eval(s, {"__builtins__": {}}, {"nan": np.nan, "inf": np.inf})
    except Exception:
        return None


# =========================
# 1) 读取全部 evaluation，不提前删行
#    返回：
#    - feas_list : 每一行是否可行
#    - F_nor     : 归一化后的 3 目标
#    - F_raw     : 原始 3 目标（给 3D 图用）
# =========================
def read_res(path_result):
    df = pd.read_csv(path_result, sep=";")

    objs_list = df["objectives"].map(parse_obj_list)

    # 先创建固定长度的原始目标数组，默认全 nan
    F_raw = np.full((len(df), 3), np.nan)

    for i, obj in enumerate(objs_list):
        if obj is not None and len(obj) >= 3:
            F_raw[i, :] = obj[:3]

    feas_list = df["is_feasible"].astype(bool).reset_index(drop=True)


    # 归一化上下界
    f1_min, f1_max = 10.35378120612736, 41.45712974153426
    f2_min, f2_max = 1628.4303363260478, 2979.759046880825
    f3_min, f3_max = 103.37531038749586, 132.57993250080827

    f1 = F_raw[:, 0]
    f2 = F_raw[:, 1]
    f3 = F_raw[:, 2]

    f1_nor = (f1 - f1_min) / (f1_max - f1_min)
    f2_nor = (f2 - f2_min) / (f2_max - f2_min)
    f3_nor = (f3 - f3_min) / (f3_max - f3_min)

    F_nor = np.column_stack([f1_nor, f2_nor, f3_nor])

    # 不删行，只把数值非法的点标成不可行
    finite_mask_raw = np.isfinite(F_raw).all(axis=1)
    finite_mask_nor = np.isfinite(F_nor).all(axis=1)
    finite_mask = finite_mask_raw & finite_mask_nor

    feas_list = (feas_list & pd.Series(finite_mask)).reset_index(drop=True)

    return feas_list, F_nor, F_raw


# =========================
# 2) Hypervolume
# =========================
def hypervolume(F):
    if F is None or len(F) == 0:
        return 0.0

    front0_idx = NonDominatedSorting().do(F)[0]
    F_nd = F[front0_idx]

    ref_point = np.array([1.1, 1.1, 1.1])
    hv = HV(ref_point=ref_point).do(F_nd)
    return hv


def hv_analysis(F, feas_idx, step=1):
    """
    每一步在当前累计 evaluation 中筛可行解
    若当前没有可行解，则 HV=0
    """
    hv_list = []

    for i in range(step - 1, len(F), step):
        feas_now = feas_idx.iloc[:i + 1].to_numpy(dtype=bool)
        F_now = F[:i + 1]
        F_feasible = F_now[feas_now]

        hv = hypervolume(F_feasible)
        hv_list.append(hv)

    return hv_list


# =========================
# 3) Feasible ratio
# =========================
def feas_ratio(feas_idx, step=1):
    feasible_ratios = []
    total_budget = len(feas_idx)

    for i in range(step - 1, len(feas_idx), step):
        feasible_count = feas_idx.iloc[:i + 1].sum()
        ratio = feasible_count / total_budget
        feasible_ratios.append(ratio)

    return feasible_ratios


# =========================
# 4) 3D 图数据拆分
#    - 红色：可行且 non-dominated
#    - 浅蓝：可行但被支配
#    - 灰色：非可行
# =========================
def split_3obj_for_plot(F_raw, feas_mask):
    feas_mask = np.asarray(feas_mask, dtype=bool)

    finite_mask = np.isfinite(F_raw).all(axis=1)
    feas_mask = feas_mask & finite_mask

    F_feas = F_raw[feas_mask]
    F_infeas = F_raw[(~feas_mask) & finite_mask]

    if len(F_feas) == 0:
        return None, None, F_infeas

    front0_idx = NonDominatedSorting().do(F_feas)[0]

    nd_mask = np.zeros(len(F_feas), dtype=bool)
    nd_mask[front0_idx] = True

    F_front0 = F_feas[nd_mask]
    F_dom = F_feas[~nd_mask]

    return F_front0, F_dom, F_infeas


def plot_3d_pareto_with_all_types(F_front0, F_dom, F_infeas, title, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 非可行解：灰色
    if F_infeas is not None and len(F_infeas) > 0:
        ax.scatter(
            F_infeas[:, 0], F_infeas[:, 1], F_infeas[:, 2],
            s=14, alpha=0.25, c="gray", label="Infeasible"
        )

    # 可行但被支配：浅蓝
    if F_dom is not None and len(F_dom) > 0:
        ax.scatter(
            F_dom[:, 0], F_dom[:, 1], F_dom[:, 2],
            s=18, alpha=0.35, c="lightblue", label="Feasible dominated"
        )

    # Pareto front：红色
    if F_front0 is not None and len(F_front0) > 0:
        ax.scatter(
            F_front0[:, 0], F_front0[:, 1], F_front0[:, 2],
            s=32, alpha=0.95, c="red", label="Pareto front"
        )

    ax.set_xlabel("f1 (raw)")
    ax.set_ylabel("f2 (raw)")
    ax.set_zlabel("f3 (raw)")
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 5) 主程序：读取三个 seed
# =========================
step = 1

cons_331, Fnor_331, Fraw_331 = read_res(PATH_random_search_331)
cons_332, Fnor_332, Fraw_332 = read_res(PATH_random_search_332)
cons_333, Fnor_333, Fraw_333 = read_res(PATH_random_search_333)

print("seed331 total evals:", len(Fnor_331), "feasible count:", int(cons_331.sum()))
print("seed332 total evals:", len(Fnor_332), "feasible count:", int(cons_332.sum()))
print("seed333 total evals:", len(Fnor_333), "feasible count:", int(cons_333.sum()))


# =========================
# 6) HV 曲线
# =========================
hv_331 = hv_analysis(Fnor_331, cons_331, step)
hv_332 = hv_analysis(Fnor_332, cons_332, step)
hv_333 = hv_analysis(Fnor_333, cons_333, step)

fea_331 = feas_ratio(cons_331, step)
fea_332 = feas_ratio(cons_332, step)
fea_333 = feas_ratio(cons_333, step)

print("len(hv_331) =", len(hv_331))
print("len(hv_332) =", len(hv_332))
print("len(hv_333) =", len(hv_333))

print("len(fea_331) =", len(fea_331))
print("len(fea_332) =", len(fea_332))
print("len(fea_333) =", len(fea_333))


# 自动截断到最短长度，避免 inhomogeneous shape 报错
hv_runs_raw = [hv_331, hv_332, hv_333]
min_len_hv = min(len(h) for h in hv_runs_raw)
hv_runs = np.array([h[:min_len_hv] for h in hv_runs_raw])

hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)

fea_runs_raw = [fea_331, fea_332, fea_333]
min_len_fea = min(len(f) for f in fea_runs_raw)
fea_runs = np.array([f[:min_len_fea] for f in fea_runs_raw])

fea_mean = fea_runs.mean(axis=0)
fea_std = fea_runs.std(axis=0)

x_hv = np.arange(1, len(hv_mean) + 1) * step
x_fea = np.arange(1, len(fea_mean) + 1) * step


# 可行率图
plt.figure(figsize=(6, 4))
plt.plot(x_fea, fea_mean, label="Mean Feasible Ratio", color="C1")
plt.fill_between(
    x_fea,
    np.maximum(fea_mean - fea_std, 0.0),
    np.minimum(fea_mean + fea_std, 1.0),
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
plt.savefig(RESULT_DIR / "Feasible_ratio_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# HV 图
plt.figure(figsize=(6, 4))
hv_lower = np.maximum(hv_mean - hv_std, 0.0)
hv_upper = hv_mean + hv_std

plt.plot(x_hv, hv_mean, label="Mean HV", color="C0")
plt.fill_between(
    x_hv,
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
plt.savefig(RESULT_DIR / "HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()


