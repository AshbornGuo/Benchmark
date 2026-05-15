from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

BASE_DIR = PROJECT_ROOT / "results" / "LayeredBeam"


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

    return np.column_stack([f1_nor, f2_nor])  


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

    max_len = max(len(r) for r in runs)
    out = np.full((len(runs), max_len), pad_value, dtype=float)
    for k, r in enumerate(runs):
        out[k, :len(r)] = r
    return out


def compute_algo_hv_stats(csv_paths: list[Path], step: int) -> tuple[np.ndarray, np.ndarray]:

    hv_runs = []
    for p in csv_paths:
        F = read_res(p)
        hv_curve = hv_analysis(F, step=step)
        hv_runs.append(hv_curve)

    hv_mat = pad_to_same_length(hv_runs, pad_value=np.nan)  # (n_runs, T_max)
    hv_mean = np.nanmean(hv_mat, axis=0)
    hv_std = np.nanstd(hv_mat, axis=0)
    return hv_mean, hv_std



def export_hv_summary_points(eval_points=(100, 200, 300, 400, 500), step=5):
    result = []

    for algo_name, paths in ALGO_CSVS.items():
        hv_at_points_all_seeds = []

        for p in paths:
            F = read_res(p)
            hv_curve = hv_analysis(F, step=step)

            hv_at_points = []
            for e in eval_points:
                if e % step != 0:
                    raise ValueError(f"{e} 不能被 step={step} 整除")

                idx = e // step - 1

                if idx < len(hv_curve):
                    hv_at_points.append(hv_curve[idx])
                else:
                    hv_at_points.append(np.nan)

            hv_at_points_all_seeds.append(hv_at_points)

        hv_at_points_all_seeds = np.array(hv_at_points_all_seeds, dtype=float)

        hv_mean = np.nanmean(hv_at_points_all_seeds, axis=0)
        hv_std = np.nanstd(hv_at_points_all_seeds, axis=0)

        row = {"Algorithm": algo_name}
        for e, m, s in zip(eval_points, hv_mean, hv_std):
            row[f"{e}"] = f"{m:.6f} ± {s:.6f}"

        result.append(row)

    df = pd.DataFrame(result)

    out_csv = BASE_DIR / "HV_summary_100_500.csv"
    df.to_csv(out_csv, index=False)

    # print(df)


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


step = 5

plt.figure(figsize=(7, 4.5))

global_max_T = 0
stats = {}


color_map = {
    "RandomSearch": "#706E6E",  
    "EGBO": "#8c564b",          
    "NSGA2": "#1f77b4",         
    "MOEAD": "#9467bd",         
    "SMSEMOA": "#d62728",       
    "EHVI": "#2ca02c",          
    "ParEGO": "#ff7f0e",        
    "MESMO": "#17becf",         
}

for algo_name, paths in ALGO_CSVS.items():

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{algo_name}] Missing CSVs:\n" + "\n".join(map(str, missing)))

    mean, std = compute_algo_hv_stats(paths, step=step)
    stats[algo_name] = (mean, std)
    global_max_T = max(global_max_T, len(mean))


for algo_name, (hv_mean, hv_std) in stats.items():
    T = len(hv_mean)
    x = np.arange(1, T + 1) * step

    hv_lower = np.maximum(hv_mean - hv_std, 0.0)
    hv_upper = hv_mean + hv_std


    x_plot = np.insert(x, 0, 0)
    hv_mean_plot = np.insert(hv_mean, 0, 0.0)
    hv_lower_plot = np.insert(hv_lower, 0, 0.0)
    hv_upper_plot = np.insert(hv_upper, 0, 0.0)
    

    color = color_map.get(algo_name, None)

    plt.plot(x_plot, hv_mean_plot, label=algo_name, color=color)
    plt.fill_between(x_plot, hv_lower_plot, hv_upper_plot, alpha=0.20, color=color)



plt.xlabel("Evaluations")
plt.ylabel("Hypervolume")


plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),   
    ncol=4,                        
    frameon=True
)


plt.grid(True)
plt.tight_layout()

out_path = BASE_DIR /  "HV_all.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()



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




export_hv_table(start_eval=55, end_eval=500, step=5)

export_hv_summary_points(eval_points=(100, 200, 300, 400, 500), step=5)