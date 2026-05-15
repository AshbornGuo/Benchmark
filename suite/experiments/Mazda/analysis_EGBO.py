

import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

from pathlib import Path

# # read csv to get reuslts of algorithms   
# PATH_EGBO_331 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed331.csv"
# PATH_EGBO_332 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed332.csv"
# PATH_EGBO_333 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EGBO/Mazda_EGBO_seed333.csv"



BASE_DIR = Path(__file__).resolve().parents[3]
RESULT_DIR = BASE_DIR / "results" / "mazda" / "SBO_EGBO"

PATH_EGBO_331 = RESULT_DIR / "Mazda_EGBO_seed331.csv"
PATH_EGBO_332 = RESULT_DIR / "Mazda_EGBO_seed332.csv"
PATH_EGBO_333 = RESULT_DIR / "Mazda_EGBO_seed333.csv"

def read_res(PATH_RESULT):

    df = pd.read_csv(PATH_RESULT,sep = ";")
    
    
    objs_list = df["objectives"].map(ast.literal_eval)
    cons_list = df["is_feasible"]

    F = np.vstack(objs_list.to_numpy())
    F = F[:, 0: 2]

    f1 = F[:, 0]
    f2 = F[:, 1]


    f1_nor = f1 - 2.0                    
    f2_nor = f2 / 74.0                    

    F_nor = np.column_stack([f1_nor, f2_nor])   

    return cons_list, F_nor


def hypervolume(F):
    if F is None or len(F) == 0:
        return 0.0
    # nondominate
    front0_idx = NonDominatedSorting().do(F)[0] # index of nondominated solution
    F_nd = F[front0_idx] # nondominated set

    # compute hypervolune
    ref_point = np.array([1.1, 0.0])
    hv = HV(ref_point=ref_point).do(F_nd)

    return hv


def hv_analysis(F, fea_inx, step = 50):
    HV = []
    for i in range(step-1, len(F), step):
        # filter feasible solutions
        cons_feasible = fea_inx[:i+1]
        F_step = F[:i+1]
        F_feasible = F_step[cons_feasible]
 

        hv = hypervolume(F_feasible)
        HV.append(hv)

    return HV


# feasible solution ratio
def feas_ratio(df, step = 50) -> list:
    feasible_ratios = []

    for i in range(step-1, len(df), step):
        feasible_ratio=df.iloc[:i+1].mean()
        feasible_ratios.append(feasible_ratio)

    return feasible_ratios



step = 1

# constaints, normalized objectives 
cons_EGBO_331, Fnor_EGBO_331 = read_res(PATH_EGBO_331)
cons_EGBO_332, Fnor_EGBO_332 = read_res(PATH_EGBO_332)
cons_EGBO_333, Fnor_EGBO_333 = read_res(PATH_EGBO_333)


# feasible rate
fea_EGBO_331 = feas_ratio(cons_EGBO_331,step)
fea_EGBO_332 = feas_ratio(cons_EGBO_332,step)
fea_EGBO_333 = feas_ratio(cons_EGBO_333,step)


# hypervolume
hv_EGBO_331 = hv_analysis(Fnor_EGBO_331, cons_EGBO_331,step)
hv_EGBO_332 = hv_analysis(Fnor_EGBO_332, cons_EGBO_332,step)
hv_EGBO_333 = hv_analysis(Fnor_EGBO_333, cons_EGBO_333,step)


# take average
hv_runs = [hv_EGBO_331,hv_EGBO_332,hv_EGBO_333]  
hv_runs = np.array(hv_runs)   
hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)


fea_runs = [fea_EGBO_331,fea_EGBO_332,fea_EGBO_333]  
fea_runs = np.array(fea_runs)   
fea_mean = fea_runs.mean(axis=0)
fea_std = fea_runs.std(axis=0)


T = len(hv_mean)
x = np.arange(1, T + 1) * step



## Mean Feasible Ratio
plt.figure(figsize=(6, 4))

plt.plot(x, fea_mean, label="Mean Feasible Ratio", color="C1")
plt.fill_between(
    x,
    fea_mean - fea_std,
    fea_mean + fea_std,
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
# plt.savefig("results/mazda/SBO_EGBO/Feasible_ratio_curve.png", dpi=300, bbox_inches="tight")
plt.savefig(RESULT_DIR / "Feasible_ratio_curve.png", dpi=300, bbox_inches="tight")
plt.close()




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
# plt.savefig("results/mazda/SBO_EGBO/HV_curve.png", dpi=300, bbox_inches="tight")
plt.savefig(RESULT_DIR / "HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()



