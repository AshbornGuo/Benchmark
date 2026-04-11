from pathlib import Path
import pandas as pd
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
BASE_DIR = PROJECT_ROOT / "results" / "TurbofanArch"

# =========================
# 归一化范围
# =========================
F1_MAX = 41.45712974153426
F1_MIN = 10.35378120612736

F2_MAX = 2979.759046880825
F2_MIN = 1628.4303363260478

F3_MAX = 132.57993250080827
F3_MIN = 103.37531038749586

REF_POINT = np.array([1.1, 1.1, 1.1])


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
#    - feas_mask : 每一行是否可用于 HV
#    - F_nor     : 归一化后的 3 目标
# =========================
def read_res(path_result: Path):
    df = pd.read_csv(path_result, sep=";")

    objs_list = df["objectives"].map(parse_obj_list)

    # 固定长度，默认全 nan，保持 evaluation 对齐
    F_raw = np.full((len(df), 3), np.nan)

    for i, obj in enumerate(objs_list):
        if obj is not None and len(obj) >= 3:
            F_raw[i, :] = np.asarray(obj[:3], dtype=float)

    f1 = F_raw[:, 0]
    f2 = F_raw[:, 1]
    f3 = F_raw[:, 2]

    f1_nor = (f1 - F1_MIN) / (F1_MAX - F1_MIN)
    f2_nor = (f2 - F2_MIN) / (F2_MAX - F2_MIN)
    f3_nor = (f3 - F3_MIN) / (F3_MAX - F3_MIN)

    F_nor = np.column_stack([f1_nor, f2_nor, f3_nor])

    finite_mask_raw = np.isfinite(F_raw).all(axis=1)
    finite_mask_nor = np.isfinite(F_nor).all(axis=1)
    finite_mask = finite_mask_raw & finite_mask_nor

    if "is_feasible" in df.columns:
        feas_col = df["is_feasible"].fillna(False).astype(bool).to_numpy()
        feas_mask = feas_col & finite_mask
    else:
        feas_mask = finite_mask

    return feas_mask, F_nor


# =========================
# 2) Hypervolume
# =========================
def hypervolume(F: np.ndarray) -> float:
    if F is None or len(F) == 0:
        return 0.0

    finite_mask = np.isfinite(F).all(axis=1)
    F = F[finite_mask]

    if len(F) == 0:
        return 0.0

    front0_idx = NonDominatedSorting().do(F)[0]
    F_nd = F[front0_idx]

    return HV(ref_point=REF_POINT).do(F_nd)


def hv_analysis(F: np.ndarray, feas_mask: np.ndarray, step: int = 5) -> np.ndarray:
    """
    每 step 个 evaluation 统计一次：
    在当前累计 evaluation 中，只取可用点做 HV
    """
    hv_list = []

    for i in range(step - 1, len(F), step):
        feas_now = feas_mask[:i + 1]
        F_now = F[:i + 1]
        F_feasible = F_now[feas_now]

        hv = hypervolume(F_feasible)
        hv_list.append(hv)

    return np.array(hv_list, dtype=float)


def pad_to_same_length(runs: list[np.ndarray], pad_value=np.nan) -> np.ndarray:
    max_len = max(len(r) for r in runs)
    out = np.full((len(runs), max_len), pad_value, dtype=float)
    for k, r in enumerate(runs):
        out[k, :len(r)] = r
    return out


def compute_algo_hv_stats(csv_paths: list[Path], step: int) -> tuple[np.ndarray, np.ndarray]:
    hv_runs = []

    for p in csv_paths:
        feas_mask, F = read_res(p)
        hv_curve = hv_analysis(F, feas_mask, step=step)
        hv_runs.append(hv_curve)

    hv_mat = pad_to_same_length(hv_runs, pad_value=np.nan)
    hv_mean = np.nanmean(hv_mat, axis=0)
    hv_std = np.nanstd(hv_mat, axis=0)

    return hv_mean, hv_std


ALGO_CSVS = {
    "RandomSearch": [
        BASE_DIR / "random_search" / "TurbofanArch_randomsearch_seed331.csv",
        BASE_DIR / "random_search" / "TurbofanArch_randomsearch_seed332.csv",
        BASE_DIR / "random_search" / "TurbofanArch_randomsearch_seed333.csv",
    ],

    "EGBO": [
        BASE_DIR / "EGBO" / "TurbofanArch_EGBO_seed331.csv",
        BASE_DIR / "EGBO" / "TurbofanArch_EGBO_seed332.csv",
        BASE_DIR / "EGBO" / "TurbofanArch_EGBO_seed333.csv",
    ],

    "NSGA2": [
        BASE_DIR / "NSGA2" / "TurbofanArch_NSGA2_seed331.csv",
        BASE_DIR / "NSGA2" / "TurbofanArch_NSGA2_seed332.csv",
        BASE_DIR / "NSGA2" / "TurbofanArch_NSGA2_seed333.csv",
    ],

    "MOEAD": [
        BASE_DIR / "MOEAD" / "TurbofanArch_MOEAD_seed331.csv",
        BASE_DIR / "MOEAD" / "TurbofanArch_MOEAD_seed332.csv",
        BASE_DIR / "MOEAD" / "TurbofanArch_MOEAD_seed333.csv",
    ],

    "SMSEMOA": [
        BASE_DIR / "SMS_EMOA" / "TurbofanArch_SMSEMOA_seed331.csv",
        BASE_DIR / "SMS_EMOA" / "TurbofanArch_SMSEMOA_seed332.csv",
        BASE_DIR / "SMS_EMOA" / "TurbofanArch_SMSEMOA_seed333.csv",
    ],

    "EHVI": [
        BASE_DIR / "SBO_qLogNEHVI" / "TurbofanArch_qLogNEHVI_seed331.csv",
        BASE_DIR / "SBO_qLogNEHVI" / "TurbofanArch_qLogNEHVI_seed332.csv",
        BASE_DIR / "SBO_qLogNEHVI" / "TurbofanArch_qLogNEHVI_seed333.csv",
    ],

    "ParEGO": [
        BASE_DIR / "SBO_qLogNParEGO" / "TurbofanArch_qLogNParEGO_seed331.csv",
        BASE_DIR / "SBO_qLogNParEGO" / "TurbofanArch_qLogNParEGO_seed332.csv",
        BASE_DIR / "SBO_qLogNParEGO" / "TurbofanArch_qLogNParEGO_seed333.csv",
    ],

    "MESMO": [
        BASE_DIR / "SBO_MESMO" / "TurbofanArch_MESMO_seed331.csv",
        BASE_DIR / "SBO_MESMO" / "TurbofanArch_MESMO_seed332.csv",
        BASE_DIR / "SBO_MESMO" / "TurbofanArch_MESMO_seed333.csv",
    ]
}

step = 5

# =========================
# 画总 HV 图
# =========================
plt.figure(figsize=(7, 4.5))
stats = {}

for algo_name, paths in ALGO_CSVS.items():
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{algo_name}] Missing CSVs:\n" + "\n".join(map(str, missing)))

    mean, std = compute_algo_hv_stats(paths, step=step)
    stats[algo_name] = (mean, std)

for algo_name, (hv_mean, hv_std) in stats.items():
    x = np.arange(1, len(hv_mean) + 1) * step
    hv_lower = np.maximum(hv_mean - hv_std, 0.0)
    hv_upper = hv_mean + hv_std

    plt.plot(x, hv_mean, label=algo_name)
    plt.fill_between(x, hv_lower, hv_upper, alpha=0.20)

plt.xlabel("Evaluations")
plt.ylabel("Hypervolume")
plt.title("Hypervolume Comparison")
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    frameon=True
)
plt.grid(True)
plt.tight_layout()

out_path = BASE_DIR / "HV_all.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")


# =========================
# 导出 HV 表
# =========================
def export_hv_table(start_eval=55, end_eval=500, step=5):
    if start_eval % step != 0 or end_eval % step != 0:
        raise ValueError("start_eval 和 end_eval 必须能被 step 整除。")

    eval_axis = np.arange(start_eval, end_eval + 1, step)
    result = {"eval_axis": eval_axis}

    for algo_name, paths in ALGO_CSVS.items():
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"[{algo_name}] Missing CSVs:\n" + "\n".join(map(str, missing)))

        hv_runs = []

        for p in paths:
            feas_mask, F = read_res(p)
            hv_curve = hv_analysis(F, feas_mask, step=step)

            hv_sample = []
            for e in eval_axis:
                idx = e // step - 1
                if idx < len(hv_curve):
                    hv_sample.append(hv_curve[idx])
                else:
                    hv_sample.append(np.nan)

            hv_runs.append(hv_sample)

        hv_runs = np.array(hv_runs, dtype=float)
        hv_mean = np.nanmean(hv_runs, axis=0)

        result[algo_name] = hv_mean

    df = pd.DataFrame(result)

    column_order = [
        "eval_axis",
        "EHVI",
        "ParEGO",
        "MESMO",
        "NSGA2",
        "MOEAD",
        "SMSEMOA",
        "EGBO",
        "RandomSearch"
    ]
    df = df[column_order]

    out_csv = BASE_DIR / "HV_all.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


export_hv_table(start_eval=55, end_eval=500, step=5)