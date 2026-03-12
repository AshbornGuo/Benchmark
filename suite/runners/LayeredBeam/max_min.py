
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

RESULT_DIR = PROJECT_ROOT / "results" / "LayeredBeam" 



PATH_RS_331 = RESULT_DIR / "randomsearch" / "LayeredBeam_RS_seed331.csv"
PATH_RS_332 = RESULT_DIR / "randomsearch" / "LayeredBeam_RS_seed332.csv"
PATH_RS_333 = RESULT_DIR / "randomsearch" / "LayeredBeam_RS_seed333.csv"

PATH_NSGA2_331 = RESULT_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed331.csv"
PATH_NSGA2_332 = RESULT_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed332.csv"
PATH_NSGA2_333 = RESULT_DIR / "NSGA2" / "LayeredBeam_NSGA2_seed333.csv"

PATH_MOEAD_331 = RESULT_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed331.csv"
PATH_MOEAD_332 = RESULT_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed332.csv"
PATH_MOEAD_333 = RESULT_DIR / "MOEAD" / "LayeredBeam_MOEAD_seed333.csv"  

PATH_SMSEMOA_331 = RESULT_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed331.csv"
PATH_SMSEMOA_332 = RESULT_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed332.csv"
PATH_SMSEMOA_333 = RESULT_DIR / "SMSEMOA" / "LayeredBeam_SMSEMOA_seed333.csv" #

PATH_EHVI_331 = RESULT_DIR / "qLogNEHVI" / "LayeredBeam_qLogNEHVI_seed331.csv"
PATH_EHVI_332 = RESULT_DIR / "qLogNEHVI" / "LayeredBeam_qLogNEHVI_seed332.csv"
PATH_EHVI_333 = RESULT_DIR / "qLogNEHVI" / "LayeredBeam_qLogNEHVI_seed333.csv"

PATH_ParEGO_331 = RESULT_DIR / "qLogNParEGO" / "LayeredBeam_qLogNParEGO_seed331.csv"
PATH_ParEGO_332 = RESULT_DIR / "qLogNParEGO" / "LayeredBeam_qLogNParEGO_seed332.csv"
PATH_ParEGO_333 = RESULT_DIR / "qLogNParEGO" / "LayeredBeam_qLogNParEGO_seed333.csv"

PATH_MESMO_331 = RESULT_DIR / "MESMO" / "LayeredBeam_MESMO_seed331.csv"
PATH_MESMO_332 = RESULT_DIR / "MESMO" / "LayeredBeam_MESMO_seed332.csv"
PATH_MESMO_333 = RESULT_DIR / "MESMO" / "LayeredBeam_MESMO_seed333.csv"

PATH_EGBO_331 = RESULT_DIR / "EGBO" / "LayeredBeam_EGBO_seed331.csv"
PATH_EGBO_332 = RESULT_DIR / "EGBO" / "LayeredBeam_EGBO_seed332.csv"
PATH_EGBO_333 = RESULT_DIR / "EGBO" / "LayeredBeam_EGBO_seed333.csv"



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
    # f2_minus = -f2
    f1_max = f1.max()
    f1_min = f1.min()    
    f2_max = f2.max()
    f2_min = f2.min()


    return [f1_max, f1_min, f2_max, f2_min]

alg_list = [PATH_RS_331, PATH_RS_332, PATH_RS_333,
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

    for p in path_list:
        f1M, f1m, f2M, f2m = read_res(p)

        f1_max = max(f1_max, f1M)
        f1_min = min(f1_min, f1m)
        f2_max = max(f2_max, f2M)
        f2_min = min(f2_min, f2m)

    return f1_max, f1_min, f2_max, f2_min

print("f1_max, f1_min, f2_max, f2_min:", max_min(alg_list))



    # f1_max = 4.691520000000004;  f1_min = 2.5315200000000093
    # f2_max = 13.638929999999998; f2_min =  -2.499999999999873