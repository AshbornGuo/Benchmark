from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

BASE_DIR = PROJECT_ROOT / "results" / "LayeredBeam"

# =========================
# 你现在的归一化范围（先沿用）
# =========================
F1_MAX = 4.691520000000004
F1_MIN = 2.5315200000000093
F2_MAX = 13.638929999999998
F2_MIN = -2.499999999999873

REF_POINT = np.array([1.1, 1.1])


def read_res(path_result: Path) -> np.ndarray:
    df = pd.read_csv(path_result, sep=";")

    objs_list = df["objectives"].map(ast.literal_eval)
    F = np.vstack(objs_list.to_numpy())

    f1 = F[:, 0]
    f2 = F[:, 1]

    f1_nor = (f1 - F1_MIN) / (F1_MAX - F1_MIN)
    f2_nor = (f2 - F2_MIN) / (F2_MAX - F2_MIN)

    return np.column_stack([f1_nor, f2_nor])  # (N, 2)


def hypervolume(F: np.ndarray) -> float:
    if F is None or len(F) == 0:
        return 0.0

    front0_idx = NonDominatedSorting().do(F)[0]
    F_nd = F[front0_idx]

    return HV(ref_point=REF_POINT).do(F_nd)


def hv_analysis(F: np.ndarray, step: int = 50) -> np.ndarray:
    hv_list = []
    for i in range(step, len(F) + 1, step):
        hv_list.append(hypervolume(F[:i]))
    return np.array(hv_list, dtype=float)


def pad_to_same_length(runs: list[np.ndarray], pad_value=np.nan) -> np.ndarray:
    """不同 run 的评估次数可能不同：用 NaN 补齐，后面用 nanmean/nanstd。"""
    max_len = max(len(r) for r in runs)
    out = np.full((len(runs), max_len), pad_value, dtype=float)
    for k, r in enumerate(runs):
        out[k, :len(r)] = r
    return out


def compute_algo_hv_stats(csv_paths: list[Path], step: int) -> tuple[np.ndarray, np.ndarray]:
    """输入某算法多个 seed CSV，输出 mean/std 曲线（按 step 聚合）。"""
    hv_runs = []
    for p in csv_paths:
        F = read_res(p)
        hv_curve = hv_analysis(F, step=step)
        hv_runs.append(hv_curve)

    hv_mat = pad_to_same_length(hv_runs, pad_value=np.nan)  # (n_runs, T_max)
    hv_mean = np.nanmean(hv_mat, axis=0)
    hv_std = np.nanstd(hv_mat, axis=0)
    return hv_mean, hv_std

##### feasibility ratio  ####
def compute_feasible_rate(csv_path: Path) -> float:
    df = pd.read_csv(csv_path, sep=";")
    feasible_count = df["is_feasible"].sum()   # True 会当作 1
    total = len(df)
    return feasible_count / total


def compute_algo_feasible_stats():
    results = {}

    for algo_name, paths in ALGO_CSVS.items():
        rates = []

        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)

            r = compute_feasible_rate(p)
            rates.append(r)

        rates = np.array(rates)

        results[algo_name] = {
            "per_seed": rates,
            "mean": rates.mean(),
            "std": rates.std()
        }

    return results


def export_feasible_table(stats):
    rows = []

    for algo, s in stats.items():
        row = {
            "Algorithm": algo,
            "Mean": s["mean"],
            "Std": s["std"]
        }

        for i, r in enumerate(s["per_seed"]):
            row[f"Seed{i+1}"] = r

        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = BASE_DIR / "Feasible_rate.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
##### feasibility ratio  ####


# =========================
# 在这里把“不同算法”的 CSV 路径都列出来
# 每个算法：一个 list，里面放多个 seeds 的 CSV
# =========================
ALGO_CSVS = {
    "RandomSearch": [
        BASE_DIR / "randomsearch" / "LayeredBeam_RS_seed331.csv",
        BASE_DIR / "randomsearch" / "LayeredBeam_RS_seed332.csv",
        BASE_DIR / "randomsearch" / "LayeredBeam_RS_seed333.csv",
    ],

    "EGBO": [
        BASE_DIR / "EGBO" / "LayeredBeam_EGBO_seed331.csv",
        BASE_DIR / "EGBO" / "LayeredBeam_EGBO_seed332.csv",
        BASE_DIR / "EGBO" / "LayeredBeam_EGBO_seed333.csv",
    ],

    "NSGA2": [
        BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed331.csv",
        BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed332.csv",
        BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed333.csv",
    ],
    
    "MOEAD": [
        BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed331.csv",
        BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed332.csv",
        BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed333.csv",
    ],

    "SMSEMOA": [
        BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed331.csv",
        BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed332.csv",
        BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed333.csv",
    ],

    "EHVI": [
        BASE_DIR / "qLogNEHVI" / "LayeredBeam_qLogNEHVI_seed331.csv",
        BASE_DIR / "qLogNEHVI" / "LayeredBeam_qLogNEHVI_seed332.csv",
        BASE_DIR / "qLogNEHVI" / "LayeredBeam_qLogNEHVI_seed333.csv",
    ],

    "ParEGO": [
        BASE_DIR / "qLogNParEGO" / "LayeredBeam_qLogNParEGO_seed331.csv",
        BASE_DIR / "qLogNParEGO" / "LayeredBeam_qLogNParEGO_seed332.csv",
        BASE_DIR / "qLogNParEGO" / "LayeredBeam_qLogNParEGO_seed333.csv",
    ],

    "MESMO": [
        BASE_DIR / "MESMO" / "LayeredBeam_MESMO_seed331.csv",
        BASE_DIR / "MESMO" / "LayeredBeam_MESMO_seed332.csv",
        BASE_DIR / "MESMO" / "LayeredBeam_MESMO_seed333.csv",
    ]

}

# 你原来 step=1 也可以，但会很慢（每次都做一次 NDS + HV）
step = 5

# =========================
# 计算并画在一张图里
# =========================
plt.figure(figsize=(7, 4.5))

global_max_T = 0
stats = {}

for algo_name, paths in ALGO_CSVS.items():
    # 防呆：路径不存在就直接报清楚
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{algo_name}] Missing CSVs:\n" + "\n".join(map(str, missing)))

    mean, std = compute_algo_hv_stats(paths, step=step)
    stats[algo_name] = (mean, std)
    global_max_T = max(global_max_T, len(mean))

# x 轴统一到最长的那个（短的曲线只画到它自己的长度）
for algo_name, (hv_mean, hv_std) in stats.items():
    T = len(hv_mean)
    x = np.arange(1, T + 1) * step

    hv_lower = np.maximum(hv_mean - hv_std, 0.0)
    hv_upper = hv_mean + hv_std

    # 手动在最前面加一个零点
    x_plot = np.insert(x, 0, 0)
    hv_mean_plot = np.insert(hv_mean, 0, 0.0)
    hv_lower_plot = np.insert(hv_lower, 0, 0.0)
    hv_upper_plot = np.insert(hv_upper, 0, 0.0)
    
    plt.plot(x_plot, hv_mean_plot, label=algo_name)
    plt.fill_between(x_plot, hv_lower_plot, hv_upper_plot, alpha=0.20)

    # plt.plot(x, hv_mean, label=f"{algo_name}")
    # plt.fill_between(x, hv_lower, hv_upper, alpha=0.20)

plt.xlabel("Evaluations")
plt.ylabel("Hypervolume")
plt.title("Hypervolume Comparison")
# plt.legend()

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),   # 往图下方移动
    ncol=4,                        # 一行放 4 个，可自己调
    frameon=True
)

# plt.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     frameon=True
# )

plt.tight_layout(rect=[0, 0, 0.82, 1])

plt.grid(True)
plt.tight_layout()

out_path = BASE_DIR /  "HV_all.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")



#############
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
            F = read_res(p)

            hv_curve = hv_analysis(F, step=step)

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



###############打印可行率
stats = compute_algo_feasible_stats()

for algo, s in stats.items():
    print(f"\n=== {algo} ===")
    for i, r in enumerate(s["per_seed"]):
        print(f"seed{i+1}: {r:.4f}")
    print(f"mean : {s['mean']:.4f}")
    print(f"std  : {s['std']:.4f}")

export_feasible_table(stats)