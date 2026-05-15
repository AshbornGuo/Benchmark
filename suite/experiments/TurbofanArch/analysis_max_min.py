from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

RESULT_DIR = PROJECT_ROOT / "results" / "TurbofanArch" 

PATH_RS_331 = RESULT_DIR / "random_search" / "TurbofanArch_randomsearch_seed331.csv"
PATH_RS_332 = RESULT_DIR / "random_search" / "TurbofanArch_randomsearch_seed332.csv"
PATH_RS_333 = RESULT_DIR / "random_search" / "TurbofanArch_randomsearch_seed333.csv"

PATH_NSGA2_331 = RESULT_DIR / "NSGA2" / "TurbofanArch_NSGA2_seed331.csv"
PATH_NSGA2_332 = RESULT_DIR / "NSGA2" / "TurbofanArch_NSGA2_seed332.csv"
PATH_NSGA2_333 = RESULT_DIR / "NSGA2" / "TurbofanArch_NSGA2_seed333.csv"

PATH_MOEAD_331 = RESULT_DIR / "MOEAD" / "TurbofanArch_MOEAD_seed331.csv"
PATH_MOEAD_332 = RESULT_DIR / "MOEAD" / "TurbofanArch_MOEAD_seed332.csv"
PATH_MOEAD_333 = RESULT_DIR / "MOEAD" / "TurbofanArch_MOEAD_seed333.csv"  

PATH_SMSEMOA_331 = RESULT_DIR / "SMS_EMOA" / "TurbofanArch_SMSEMOA_seed331.csv"
PATH_SMSEMOA_332 = RESULT_DIR / "SMS_EMOA" / "TurbofanArch_SMSEMOA_seed332.csv"
PATH_SMSEMOA_333 = RESULT_DIR / "SMS_EMOA" / "TurbofanArch_SMSEMOA_seed333.csv"

PATH_EHVI_331 = RESULT_DIR / "SBO_qLogNEHVI" / "TurbofanArch_qLogNEHVI_seed331.csv"
PATH_EHVI_332 = RESULT_DIR / "SBO_qLogNEHVI" / "TurbofanArch_qLogNEHVI_seed332.csv"
PATH_EHVI_333 = RESULT_DIR / "SBO_qLogNEHVI" / "TurbofanArch_qLogNEHVI_seed333.csv"

PATH_ParEGO_331 = RESULT_DIR / "SBO_qLogNParEGO" / "TurbofanArch_qLogNParEGO_seed331.csv"
PATH_ParEGO_332 = RESULT_DIR / "SBO_qLogNParEGO" / "TurbofanArch_qLogNParEGO_seed332.csv"
PATH_ParEGO_333 = RESULT_DIR / "SBO_qLogNParEGO" / "TurbofanArch_qLogNParEGO_seed333.csv"

PATH_MESMO_331 = RESULT_DIR / "SBO_MESMO" / "TurbofanArch_MESMO_seed331.csv"
PATH_MESMO_332 = RESULT_DIR / "SBO_MESMO" / "TurbofanArch_MESMO_seed332.csv"
PATH_MESMO_333 = RESULT_DIR / "SBO_MESMO" / "TurbofanArch_MESMO_seed333.csv"

PATH_EGBO_331 = RESULT_DIR / "EGBO" / "TurbofanArch_EGBO_seed331.csv"
PATH_EGBO_332 = RESULT_DIR / "EGBO" / "TurbofanArch_EGBO_seed332.csv"
PATH_EGBO_333 = RESULT_DIR / "EGBO" / "TurbofanArch_EGBO_seed333.csv"


def read_res(path_result):
    df = pd.read_csv(path_result, sep=";")


    df_feas = df[df["is_feasible"] == True].copy()


    if df_feas.empty:
        return None


    objs_list = df_feas["objectives"].map(ast.literal_eval)
    F = np.vstack(objs_list.to_numpy())

    # 只取前两个目标
    f1 = F[:, 0]
    f2 = F[:, 1]
    f3 = F[:, 2]

    f1_max = f1.max()
    f1_min = f1.min()
    f2_max = f2.max()
    f2_min = f2.min()
    f3_max = f3.max()
    f3_min = f3.min()

    return [f1_max, f1_min, f2_max, f2_min, f3_max, f3_min]


alg_list = [
    PATH_RS_331, PATH_RS_332, PATH_RS_333,
    PATH_NSGA2_331, PATH_NSGA2_332, PATH_NSGA2_333,
    PATH_MOEAD_331, PATH_MOEAD_332, PATH_MOEAD_333,
    PATH_SMSEMOA_331, PATH_SMSEMOA_332, PATH_SMSEMOA_333,
    PATH_EHVI_331, PATH_EHVI_332, PATH_EHVI_333,
    PATH_ParEGO_331, PATH_ParEGO_332, PATH_ParEGO_333,
    PATH_MESMO_331, PATH_MESMO_332, PATH_MESMO_333,
    PATH_EGBO_331, PATH_EGBO_332, PATH_EGBO_333,
]


def max_min(path_list):
    f1_max = -np.inf
    f1_min =  np.inf
    f2_max = -np.inf
    f2_min =  np.inf
    f3_max = -np.inf
    f3_min =  np.inf

    found_feasible = False

    for p in path_list:
        res = read_res(p)


        if res is None:
            print(f"[WARNING] No feasible solutions in: {p}")
            continue

        found_feasible = True
        f1M, f1m, f2M, f2m, f3M, f3m = res

        f1_max = max(f1_max, f1M)
        f1_min = min(f1_min, f1m)
        f2_max = max(f2_max, f2M)
        f2_min = min(f2_min, f2m)
        f3_max = max(f3_max, f3M)
        f3_min = min(f3_min, f3m)

    if not found_feasible:
        raise ValueError("No feasible solutions found in any result file.")

    return f1_max, f1_min, f2_max, f2_min, f3_max, f3_min


print("f1_max, f1_min, f2_max, f2_min, f3_max, f3_min:", max_min(alg_list))



    # f1_min, f1_max = 10.35378120612736, 41.45712974153426
    # f2_min, f2_max = 1628.4303363260478, 2979.759046880825
    # f3_min, f3_max = 103.37531038749586, 132.57993250080827


