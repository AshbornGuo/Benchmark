
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

RESULT_DIR = PROJECT_ROOT / "results" / "LayeredBeam" / "qLogNParEGO"


PATH_qLogNParEGO_331 = RESULT_DIR / "LayeredBeam_qLogNParEGO_seed331.csv"
PATH_qLogNParEGO_332 = RESULT_DIR / "LayeredBeam_qLogNParEGO_seed332.csv"
PATH_qLogNParEGO_333 = RESULT_DIR / "LayeredBeam_qLogNParEGO_seed333.csv"


def read_res(PATH_RESULT):

    df = pd.read_csv(PATH_RESULT, sep=";")

    objs_list = df["objectives"].map(ast.literal_eval)


    F = np.vstack(objs_list.to_numpy())

    f1 = F[:, 0]
    f2 = F[:, 1]
    
    # normalized 
    f1_max = 4.691520000000004;  f1_min = 2.5315200000000093
    f2_max = 13.638929999999998; f2_min =  -2.499999999999873
   
    f1_nor = (f1 - f1_min) / ( f1_max - f1_min) 
 
    f2_nor = (f2 - f2_min) / (f2_max - f2_min)   
      

    F_nor = np.column_stack([f1_nor, f2_nor])   # shape: (N, 2)
    return F_nor





def hypervolume(F):
    if F is None or len(F) == 0:
        return 0.0
    # nondominate
    front0_idx = NonDominatedSorting().do(F)[0] # index of nondominated solution
    F_nd = F[front0_idx] # nondominated set

    # compute hypervolune
    ref_point = np.array([1.1, 1.1])
    hv = HV(ref_point=ref_point).do(F_nd)

    return hv


def hv_analysis(F, step):
    hv_list = []

    for i in range(step, len(F) + 1, step):
        hv = hypervolume(F[:i])   
        hv_list.append(hv)

    return hv_list

step = 1

# constaints, normalized objectives 
Fnor_qLogNParEGO_331 = read_res(PATH_qLogNParEGO_331)
Fnor_qLogNParEGO_332 = read_res(PATH_qLogNParEGO_332)
Fnor_qLogNParEGO_333 = read_res(PATH_qLogNParEGO_333)


# hypervolume
hv_qLogNParEGO_331 = hv_analysis(Fnor_qLogNParEGO_331, step)
hv_qLogNParEGO_332 = hv_analysis(Fnor_qLogNParEGO_332, step)
hv_qLogNParEGO_333 = hv_analysis(Fnor_qLogNParEGO_333, step)


# take average
hv_runs = [hv_qLogNParEGO_331, hv_qLogNParEGO_332, hv_qLogNParEGO_333]  

hv_runs = np.array(hv_runs)   # shape: (5, T)
hv_mean = hv_runs.mean(axis=0)
hv_std = hv_runs.std(axis=0)


T = len(hv_mean)
x = np.arange(1, T + 1) * step


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
plt.savefig("results/LayeredBeam/qLogNParEGO/HV_curve.png", dpi=300, bbox_inches="tight")
plt.close()

