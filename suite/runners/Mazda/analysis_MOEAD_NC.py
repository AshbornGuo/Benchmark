

import pandas as pd
import numpy as np
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

# read csv to get reuslts of algorithms  
PATH_MOEAD_NC_331 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_NC_seed331.csv"
PATH_MOEAD_NC_332 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_NC_seed332.csv"
PATH_MOEAD_NC_333 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_NC_seed333.csv"
PATH_MOEAD_NC_334 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_NC_seed334.csv"
PATH_MOEAD_NC_335 = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/MOEAD/Mazda_MOEAD_NC_seed335.csv"


def read_res(PATH_RESULT):
    # read the results CSV file
    df = pd.read_csv(PATH_RESULT,sep = ";")

    # 1) 把字符串 "[...]" 安全解析成 Python list
    objs_list = df["objectives"].map(ast.literal_eval)
    cons_list = df["constraints"].map(ast.literal_eval)
    fea_list = df["is_feasible"]

    # 2) 变成 (N, D) 的 ndarray
    F = np.vstack(objs_list.to_numpy())

    # 3) 取前两个目标
    f1 = F[:, 0]
    f2 = F[:, 1]

    return cons_list, fea_list, f1, f2


# constaints, normalized objectives 
cons_MOEAD_NC_331, Fea_MOEAD_331, f1_331, f2_331 = read_res(PATH_MOEAD_NC_331)
cons_MOEAD_NC_332, Fea_MOEAD_332, f1_332, f2_332 = read_res(PATH_MOEAD_NC_332)
cons_MOEAD_NC_333, Fea_MOEAD_333, f1_333, f2_333 = read_res(PATH_MOEAD_NC_333)
cons_MOEAD_NC_334, Fea_MOEAD_334, f1_334, f2_334 = read_res(PATH_MOEAD_NC_334)
cons_MOEAD_NC_335, Fea_MOEAD_335, f1_335, f2_335 = read_res(PATH_MOEAD_NC_335)


def cv(Fea_MOEAD_X, cons_MOEAD_NC_X):
    row_neg_sums = []

    for flag, cons_list in zip(Fea_MOEAD_X, cons_MOEAD_NC_X):
        if not flag:  # 只看 False 行
            neg_sum = sum(x for x in cons_list if x < 0)
            row_neg_sums.append(neg_sum)

    # 对每一行的负数和取平均
    avg_neg_sum = sum(row_neg_sums) / len(row_neg_sums) if row_neg_sums else 0.0

    # for i in f1_331:


    return avg_neg_sum

cv_331 = cv(Fea_MOEAD_331, cons_MOEAD_NC_331)
cv_332 = cv(Fea_MOEAD_332, cons_MOEAD_NC_332)
cv_333 = cv(Fea_MOEAD_333, cons_MOEAD_NC_333)
cv_334 = cv(Fea_MOEAD_334, cons_MOEAD_NC_334)
cv_335 = cv(Fea_MOEAD_335, cons_MOEAD_NC_335)

print("average constraints violation:",(cv_331+cv_332+cv_333+cv_334+cv_335)/5)
print("avearge obj1 of infeasible solution:",(np.average(f1_331)+np.average(f1_332)+np.average(f1_333)+np.average(f1_334)+np.average(f1_335))/5)
print("avearge obj2 of infeasible solution:",(np.average(f2_331)+np.average(f2_332)+np.average(f2_333)+np.average(f2_334)+np.average(f2_335))/5)