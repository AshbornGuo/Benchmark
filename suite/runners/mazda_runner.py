import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from suite.utils import save_result, Timer

SHEET_NAME = "Explain_DV_and_Const."


# ===================== 你需要改这里（默认3个路径） =====================
DEFAULT_EXCEL_PATH = r"C:/Users/guoji/Desktop/graduate project/codes/benchmarks/Mazda_Bechmark/Mazda_CdMOBP/Info_test.xlsx"
DEFAULT_EVAL_EXE   = r"C:/Users/guoji/Desktop/graduate project/codes/benchmarks/Mazda_Bechmark/Mazda_CdMOBP/Mazda_CdMOBP/bin/win64/mazda_mop.exe"
DEFAULT_ROOT_DIR   = r"C:/Users/guoji/Desktop/graduate project/codes/benchmarks/Mazda_Bechmark/Mazda_CdMOBP"
# ================================================================


@dataclass
class DVInfo:
    car_model: str
    var_name: str
    discrete_values: List[float]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower()
        for lc, orig in lower_map.items():
            if cand_l in lc:
                return orig
    raise KeyError(f"Cannot find a column among candidates={candidates}. Available={cols}")


def parse_discrete_volume(cell) -> List[float]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    vals = [float(x) for x in nums]
    seen = set()
    uniq = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def extract_discrete_map(excel_path: str, sheet_name: str = SHEET_NAME) -> Tuple[List[DVInfo], pd.DataFrame]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    car_col = _find_col(df, ["Car Model", "CarModel"])
    var_col = _find_col(df, ["Design Variable", "DesignVariable"])
    dv_col  = _find_col(df, ["Discrete Volume", "DiscreteVolume"])

    tmp = df[[car_col, var_col, dv_col]].copy()
    tmp = tmp.dropna(subset=[var_col, dv_col])
    tmp[var_col] = tmp[var_col].astype(str).str.strip()
    tmp = tmp[tmp[var_col].str.len() > 0]

    infos: List[DVInfo] = []
    for _, row in tmp.iterrows():
        car = str(row[car_col]).strip() if not pd.isna(row[car_col]) else ""
        var = str(row[var_col]).strip()
        dvals = parse_discrete_volume(row[dv_col])
        if len(dvals) == 0:
            continue
        infos.append(DVInfo(car_model=car, var_name=var, discrete_values=dvals))

    return infos, df


def sample_random_candidates(infos: List[DVInfo], n: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = len(infos)
    X = np.empty((n, D), dtype=float)
    for j, info in enumerate(infos):
        choices = np.array(info.discrete_values, dtype=float)
        X[:, j] = rng.choice(choices, size=n, replace=True)
    return X


def write_pop_vars(work_dir: str, X: np.ndarray) -> str:
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "pop_vars_eval.txt")
    np.savetxt(path, X, delimiter="\t", fmt="%.10g")
    return path


def read_tab_matrix(path: str) -> np.ndarray:
    A = np.loadtxt(path, delimiter="\t")
    if A.ndim == 1:
        A = A[None, :]
    return A


def run_evaluator(exe_path: str, work_dir: str) -> None:
    subprocess.run([exe_path, work_dir], check=True)


def pick_best_candidate(F: np.ndarray, G: np.ndarray) -> Tuple[int, bool, float]:
    """
    可行优先（g<=0）
    - 可行：min f1；若 f1 相同，max f2
    - 不可行：min violation = sum(max(0,g))
    """
    feasible = np.all(G <= 0, axis=1)
    violation = np.maximum(G, 0).sum(axis=1)

    if np.any(feasible):
        feas_idx = np.where(feasible)[0]
        f1 = F[feas_idx, 0]
        best_f1 = np.min(f1)
        cand = feas_idx[f1 == best_f1]
        if cand.size == 1:
            idx = int(cand[0])
        else:
            f2 = F[cand, 1]
            idx = int(cand[np.argmax(f2)])
        return idx, True, float(violation[idx])
    else:
        idx = int(np.argmin(violation))
        return idx, False, float(violation[idx])


def run_mazda(suite, problem, algo, seed, budget, **kwargs):
    """
    suite: "mazda"
    problem: 你可以先写死一个字符串，比如 "CdMOBP"
    algo: 例如 "random_search"
    seed: 随机种子
    budget: 你想评估多少次（总候选点数）
    kwargs:
      - excel_path
      - eval_exe
      - root_dir
      - batch_size (默认 50)
    """

    # 允许你从 run_all.py 传路径覆盖默认值
    excel_path = kwargs.get("excel_path", DEFAULT_EXCEL_PATH)
    eval_exe   = kwargs.get("eval_exe",   DEFAULT_EVAL_EXE)
    root_dir   = kwargs.get("root_dir",   DEFAULT_ROOT_DIR)
    batch_size = int(kwargs.get("batch_size", 50))

    # work_dir 放在你 suite 工程里，避免污染原 benchmark 目录
    work_dir = os.path.join("third_party", "mazda_work")
    os.makedirs(work_dir, exist_ok=True)

    # 1) 读取离散变量定义
    infos, _ = extract_discrete_map(excel_path, SHEET_NAME)
    D = len(infos)
    if D == 0:
        raise RuntimeError("没有从 Excel 里解析出离散变量（D=0），请检查 SHEET_NAME/列名/Excel路径")

    # 2) 随机搜索（按 budget 次评估）
    rounds = int(np.ceil(budget / batch_size))

    best_global = None  # (key, record)
    # key: (0, f1, -f2) for feasible; (1, viol, f1) for infeasible

    last_stdout = ""
    last_stderr = ""

    with Timer() as t:
        for r in range(rounds):
            n_this = batch_size if (r < rounds - 1) else (budget - batch_size * (rounds - 1))
            if n_this <= 0:
                break

            X = sample_random_candidates(infos, n=n_this, seed=seed + r)
            write_pop_vars(work_dir, X)

            try:
                run_evaluator(eval_exe, work_dir)
            except subprocess.CalledProcessError as e:
                # exe 跑崩了，直接记失败
                out_path = f"results/{suite}/{problem}/{algo}/seed_{seed}.json"
                save_result({
                    "suite": suite,
                    "problem": problem,
                    "algorithm": algo,
                    "seed": seed,
                    "budget": budget,
                    "status": "failed",
                    "time_sec": t.elapsed,
                    "error": f"mazda exe failed: {e}",
                    "work_dir": work_dir,
                    "excel_path": excel_path,
                    "eval_exe": eval_exe,
                    "D": D,
                }, out_path)
                return

            # Mazda exe 输出文件（它会写到 work_dir）
            F_path = os.path.join(work_dir, "pop_objs_eval.txt")
            G_path = os.path.join(work_dir, "pop_cons_eval.txt")

            F = read_tab_matrix(F_path)
            G = read_tab_matrix(G_path)

            idx, feas, viol = pick_best_candidate(F, G)
            f1 = float(F[idx, 0])
            f2 = float(F[idx, 1])
            x = X[idx].copy()

            if feas:
                key = (0, f1, -f2)
            else:
                key = (1, viol, f1)

            record = {
                "round": r,
                "n_this": n_this,
                "feasible": feas,
                "violation": viol,
                "f1": f1,
                "f2": f2,
                "x": x.tolist(),
            }

            if best_global is None or key < best_global[0]:
                best_global = (key, record)

    # 3) 保存最终结果
    out_path = f"results/{suite}/{problem}/{algo}/seed_{seed}.json"

    if best_global is None:
        save_result({
            "suite": suite,
            "problem": problem,
            "algorithm": algo,
            "seed": seed,
            "budget": budget,
            "status": "failed",
            "time_sec": t.elapsed,
            "reason": "no evaluation results",
            "work_dir": work_dir,
            "excel_path": excel_path,
            "eval_exe": eval_exe,
            "D": D,
        }, out_path)
        return

    _, best = best_global
    save_result({
        "suite": suite,
        "problem": problem,
        "algorithm": algo,
        "seed": seed,
        "budget": budget,
        "status": "ok",
        "time_sec": t.elapsed,
        "D": D,
        "work_dir": work_dir,
        "excel_path": excel_path,
        "eval_exe": eval_exe,
        # “多目标输出”
        "best": best,
        # 额外提示：你如果想画 pareto 或存全量 F/G，以后再加
        "note": "目前只保存可行优先规则选出的best；如需保存每轮/全量F,G可再扩展",
    }, out_path)
