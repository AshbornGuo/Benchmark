from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
BASE_DIR = PROJECT_ROOT / "results" / "LayeredBeam"


def read_algorithm_time(path_result: Path, start_row: int = 55, step: int = 5) -> np.ndarray:
    df = pd.read_csv(path_result, sep=";")
    times = df["algorithm_time"].astype(float).to_numpy()
    sampled = times[start_row - 1 :: step]
    return sampled


def read_moead_algorithm_time(path_seed: Path, path_gen: Path) -> np.ndarray:
    seed_df = pd.read_csv(path_seed, sep=";")
    gen_df = pd.read_csv(path_gen, sep=";")

    eval_times = seed_df["evaluation_time"].astype(float).to_numpy()
    gen_times = gen_df["gen_time"].astype(float).to_numpy()

    result = []

    for g in range(1, len(gen_times)):
        start = (g + 9) * 5
        end = start + 5

        block = eval_times[start:end]

        if len(block) < 5:
            break

        eval_batch_time = np.max(block)
        algo_time = gen_times[g] - eval_batch_time

        result.append(algo_time)

    return np.array(result)


def compute_moead_time_stats(seed_paths: list[Path], gen_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    time_runs = []
    for seed_p, gen_p in zip(seed_paths, gen_paths):
        t = read_moead_algorithm_time(seed_p, gen_p)
        time_runs.append(t)

    time_mat = pad_to_same_length(time_runs, pad_value=np.nan)
    time_mean = np.nanmean(time_mat, axis=0)
    time_std = np.nanstd(time_mat, axis=0)
    return time_mean, time_std


def read_ea_algorithm_time(path_seed: Path, path_gen: Path) -> np.ndarray:
    seed_df = pd.read_csv(path_seed, sep=";")
    gen_df = pd.read_csv(path_gen, sep=";")

    eval_times = seed_df["evaluation_time"].astype(float).to_numpy()
    gen_times = gen_df["gen_time"].astype(float).to_numpy()

    result = []

    for g in range(1, len(gen_times)):
        start = g * 50
        end = start + 50

        block = eval_times[start:end]

        if len(block) < 50:
            break

        avg_list = []
        for i in range(0, 50, 5):
            avg_list.append(np.mean(block[i:i+5]))

        eval_sum = sum(avg_list)
        algo_time = gen_times[g] - eval_sum
        point = algo_time / 10.0

        result.extend([point] * 10)

    return np.array(result)


def read_randomsearch_algorithm_time(path_seed: Path, start_row: int = 55, step: int = 5) -> np.ndarray:
    seed_df = pd.read_csv(path_seed, sep=";")
    n_eval = len(seed_df)
    x = np.arange(start_row, n_eval + 1, step)
    return np.zeros(len(x), dtype=float)


def pad_to_same_length(runs: list[np.ndarray], pad_value=np.nan) -> np.ndarray:
    max_len = max(len(r) for r in runs)
    out = np.full((len(runs), max_len), pad_value, dtype=float)
    for k, r in enumerate(runs):
        out[k, :len(r)] = r
    return out


def compute_algo_time_stats(
    csv_paths: list[Path],
    start_row: int = 55,
    sample_step: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    time_runs = []
    for p in csv_paths:
        t = read_algorithm_time(p, start_row=start_row, step=sample_step)
        time_runs.append(t)

    time_mat = pad_to_same_length(time_runs, pad_value=np.nan)
    time_mean = np.nanmean(time_mat, axis=0)
    time_std = np.nanstd(time_mat, axis=0)
    return time_mean, time_std


def compute_ea_time_stats(seed_paths: list[Path], gen_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    time_runs = []
    for seed_p, gen_p in zip(seed_paths, gen_paths):
        t = read_ea_algorithm_time(seed_p, gen_p)
        time_runs.append(t)

    time_mat = pad_to_same_length(time_runs, pad_value=np.nan)
    time_mean = np.nanmean(time_mat, axis=0)
    time_std = np.nanstd(time_mat, axis=0)
    return time_mean, time_std


def compute_randomsearch_time_stats(
    seed_paths: list[Path],
    start_row: int = 55,
    step: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    time_runs = []
    for seed_p in seed_paths:
        t = read_randomsearch_algorithm_time(seed_p, start_row=start_row, step=step)
        time_runs.append(t)

    time_mat = pad_to_same_length(time_runs, pad_value=np.nan)
    time_mean = np.nanmean(time_mat, axis=0)
    time_std = np.nanstd(time_mat, axis=0)
    return time_mean, time_std


SBO_CSVS = {
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
    ],
}

EA_CSVS = {
    "NSGA2": {
        "seed": [
            BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed331.csv",
            BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed332.csv",
            BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed333.csv",
        ],
        "gen": [
            BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_gentime331.csv",
            BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_gentime332.csv",
            BASE_DIR / "NSGA2" / "LayeredBeam_NSGA2_gentime333.csv",
        ],
    },
    "MOEAD": {
        "seed": [
            BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed331.csv",
            BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed332.csv",
            BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed333.csv",
        ],
        "gen": [
            BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_gentime331.csv",
            BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_gentime332.csv",
            BASE_DIR / "MOEAD" / "LayeredBeam_MOEAD_gentime333.csv",
        ],
    },
    "SMSEMOA": {
        "seed": [
            BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed331.csv",
            BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed332.csv",
            BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed333.csv",
        ],
        "gen": [
            BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_gentime331.csv",
            BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_gentime332.csv",
            BASE_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_gentime333.csv",
        ],
    },
    "RandomSearch": {
        "seed": [
            BASE_DIR / "randomsearch" / "LayeredBeam_RS_seed331.csv",
            BASE_DIR / "randomsearch" / "LayeredBeam_RS_seed332.csv",
            BASE_DIR / "randomsearch" / "LayeredBeam_RS_seed333.csv",
        ],
    },
}

start_row = 55
sample_step = 5

plt.figure(figsize=(7, 4.5))

stats = {}

# SBO
for algo_name, paths in SBO_CSVS.items():
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{algo_name}] Missing CSVs:\n" + "\n".join(map(str, missing)))

    mean, std = compute_algo_time_stats(paths, start_row=start_row, sample_step=sample_step)
    stats[algo_name] = (mean, std)

# EA
for algo_name, files in EA_CSVS.items():
    seed_paths = files["seed"]
    gen_paths = files.get("gen", [])

    missing = [p for p in seed_paths + gen_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{algo_name}] Missing CSVs:\n" + "\n".join(map(str, missing)))

    if algo_name == "MOEAD":
        mean, std = compute_moead_time_stats(seed_paths, gen_paths)
    elif algo_name == "RandomSearch":
        mean, std = compute_randomsearch_time_stats(seed_paths, start_row=start_row, step=sample_step)
    else:
        mean, std = compute_ea_time_stats(seed_paths, gen_paths)

    stats[algo_name] = (mean, std)

# 这里新增：只保存每个算法的平均时间
algo_names = []
algo_time_means = []

for algo_name, (time_mean, time_std) in stats.items():
    algo_names.append(algo_name)
    algo_time_means.append(time_mean.tolist())

# 如果你还想要 evaluation 轴，也可以一起存
eval_axis = np.arange(
    start_row,
    start_row + len(algo_time_means[0]) * sample_step,
    sample_step
)


for algo_name, (time_mean, time_std) in stats.items():
    T = len(time_mean)
    x = np.arange(start_row, start_row + T * sample_step, sample_step)

    eps = 1e-4
    time_mean_plot = np.maximum(time_mean, eps)
    lower = np.maximum(time_mean - time_std, eps)
    upper = np.maximum(time_mean + time_std, eps)

    plt.plot(x, time_mean_plot, label=f"{algo_name} (mean)")
    plt.fill_between(x, lower, upper, alpha=0.20)

plt.yscale("log")
plt.xlabel("Evaluations")
plt.ylabel("Algorithm Time (s)")
plt.title("Algorithm Time vs Evaluations (Multiple Algorithms)")
plt.legend()
plt.grid(True)
plt.tight_layout()

out_path = BASE_DIR / "randomsearch" / "alg_time_curve2.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")


print("\n algo_names:", algo_names)
print("\n algo_time_means:", algo_time_means)
print("\n eval_axis:", eval_axis)

import pandas as pd

# 每一列是一个算法
df = pd.DataFrame({name: values for name, values in zip(algo_names, algo_time_means)})

# 插入第一列 eval_axis
df.insert(0, "eval_axis", eval_axis)

# 保存 CSV
csv_path = BASE_DIR / "algo_time_all.csv"
df.to_csv(csv_path, index=False)

print("Saved:", csv_path)





