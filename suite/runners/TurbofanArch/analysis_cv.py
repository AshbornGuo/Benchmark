import pandas as pd
import numpy as np
import ast

PATH_RS_331 = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/random_search/TurbofanArch_randomsearch_seed331.csv"
PATH_RS_332 = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/random_search/TurbofanArch_randomsearch_seed332.csv"
PATH_RS_333 = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/random_search/TurbofanArch_randomsearch_seed333.csv"

paths = [PATH_RS_331, PATH_RS_332, PATH_RS_333]

all_objectives = []
all_negative_constraint_sums = []

for path in paths:
    df = pd.read_csv(path, sep=';')
    
    # 先把布尔列统一转成真正的 bool
    df["is_valid"] = df["is_valid"].astype(str).str.strip().str.lower() == "true"
    df["is_feasible"] = df["is_feasible"].astype(str).str.strip().str.lower() == "true"
    
    # 只保留 is_valid=True 且 is_feasible=False 的行
    df_filtered = df[(df["is_valid"] == True) & (df["is_feasible"] == False)]
    
    # 处理 objectives
    obj_list = df_filtered["objectives"].apply(ast.literal_eval)
    all_objectives.extend(obj_list.tolist())
    
    # 处理 constraints
    cons_list = df_filtered["constraints"].apply(ast.literal_eval)
    for cons in cons_list:
        neg_sum = sum(x for x in cons if x > 0)
        all_negative_constraint_sums.append(neg_sum)

# objectives 每一维平均
all_objectives = np.array(all_objectives, dtype=float)
mean_objectives = all_objectives.mean(axis=0)

# constraints 负值和的平均
mean_constraints = np.mean(all_negative_constraint_sums)

print("筛选后，三个csv合并的 objectives 每一维平均值:", mean_objectives)
print("筛选后，三个csv合并的 constraints 中所有负值之和的平均值:", mean_constraints)