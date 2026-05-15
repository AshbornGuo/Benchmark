from pathlib import Path
import pandas as pd
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt


PROJECT_ROOT = Path.cwd()
RESULT_DIR = PROJECT_ROOT / "results" / "TurbofanArch" / "SBO_qLogNParEGO"

PATH_qLogNParEGO_331 = RESULT_DIR / "TurbofanArch_qLogNParEGO_seed331.csv"
PATH_qLogNParEGO_332 = RESULT_DIR / "TurbofanArch_qLogNParEGO_seed332.csv"
PATH_qLogNParEGO_333 = RESULT_DIR / "TurbofanArch_qLogNParEGO_seed333.csv"


def parse_obj_list(s):
    if pd.isna(s):
        return None

    s = str(s).strip()

    try:
        return eval(s, {"__builtins__": {}}, {"nan": np.nan, "inf": np.inf})
    except Exception:
        return None


def read_res(path_result):
    df = pd.read_csv(path_result, sep=";")

    objs_list = df["objectives"].map(parse_obj_list)

    F_raw = np.full((len(df), 3), np.nan)

    for i, obj in enumerate(objs_list):
        if obj is not None and len(obj) >= 3:
            F_raw[i, :] = obj[:3]

    feas_list = df["is_feasible"].astype(bool).reset_index(drop=True)


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

    finite_mask_raw = np.isfinite(F_raw).all(axis=1)
    finite_mask_nor = np.isfinite(F_nor).all(axis=1)
    finite_mask = finite_mask_raw & finite_mask_nor

    feas_list = (feas_list & pd.Series(finite_mask)).reset_index(drop=True)

    return feas_list, F_nor, F_raw


def hypervolume(F):
    if F is None or len(F) == 0:
        return 0.0

    front0_idx = NonDominatedSorting().do(F)[0]
    F_nd = F[front0_idx]

    ref_point = np.array([1.1, 1.1, 1.1])
    hv = HV(ref_point=ref_point).do(F_nd)
    return hv


def hv_analysis(F, feas_idx, step=1):

    hv_list = []

    for i in range(step - 1, len(F), step):
        feas_now = feas_idx.iloc[:i + 1].to_numpy(dtype=bool)
        F_now = F[:i + 1]
        F_feasible = F_now[feas_now]

        hv = hypervolume(F_feasible)
        hv_list.append(hv)

    return hv_list


def feas_ratio(feas_idx, step=1):
    feasible_ratios = []
    total_budget = len(feas_idx)

    for i in range(step - 1, len(feas_idx), step):
        feasible_count = feas_idx.iloc[:i + 1].sum()
        ratio = feasible_count / total_budget
        feasible_ratios.append(ratio)

    return feasible_ratios




step = 1

cons_331, Fnor_331, Fraw_331 = read_res(PATH_qLogNParEGO_331)
cons_332, Fnor_332, Fraw_332 = read_res(PATH_qLogNParEGO_332)
cons_333, Fnor_333, Fraw_333 = read_res(PATH_qLogNParEGO_333)


hv_331 = hv_analysis(Fnor_331, cons_331, step)
hv_332 = hv_analysis(Fnor_332, cons_332, step)
hv_333 = hv_analysis(Fnor_333, cons_333, step)

fea_331 = feas_ratio(cons_331, step)
fea_332 = feas_ratio(cons_332, step)
fea_333 = feas_ratio(cons_333, step)


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
