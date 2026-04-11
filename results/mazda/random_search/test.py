import pandas as pd
import ast
import os
import numpy as np

base_dir = r"results/mazda/random_search"

files = [
    "Mazda_RS_seed331.csv",
    "Mazda_RS_seed332.csv",
    "Mazda_RS_seed333.csv",
]

def to_bool(x):
    return str(x).strip().lower() == "true"

def parse_list(s):
    return ast.literal_eval(s)

results = []

for file_name in files:
    file_path = os.path.join(base_dir, file_name)
    df = pd.read_csv(file_path, sep=';')

    # 筛选 is_feasible == False
    sub_df = df.loc[~df["is_feasible"].apply(to_bool)].copy()

    # 解析 objectives 列
    obj_array = np.array(sub_df["objectives"].apply(parse_list).tolist(), dtype=float)

    # 对每个目标分别在所有行上取平均
    mean_objectives = obj_array.mean(axis=0)

    results.append({
        "file": file_name,
        "num_rows": len(sub_df),
        "mean_obj_1": mean_objectives[0],
        "mean_obj_2": mean_objectives[1],
        # "mean_obj_3": mean_objectives[2],
    })

result_df = pd.DataFrame(results)
print(result_df)

# 更清楚的打印
for row in results:
    print(f"{row['file']}:    {row['mean_obj_1']}, {row['mean_obj_2']}")
    # print(f"  mean objectives = []")
    print()
##############################
import pandas as pd
import ast
import os

base_dir = r"results/mazda/random_search"

files = [
    "Mazda_RS_seed331.csv",
    "Mazda_RS_seed332.csv",
    "Mazda_RS_seed333.csv",
]

def to_bool(x):
    return str(x).strip().lower() == "true"

def row_negative_constraint_sum(constraint_str):
    vals = ast.literal_eval(constraint_str)
    return sum(v for v in vals if v < 0)

results = []

for file_name in files:
    file_path = os.path.join(base_dir, file_name)
    df = pd.read_csv(file_path, sep=';')

    # 只筛选 is_feasible == False
    sub_df = df.loc[~df["is_feasible"].apply(to_bool)].copy()

    # 每一行 constraints 中所有 < 0 的值先求和
    sub_df["row_negative_constraint_sum"] = sub_df["constraints"].apply(row_negative_constraint_sum)

    # 再对这些行取平均
    avg_negative_violation = sub_df["row_negative_constraint_sum"].mean()

    results.append({
        "file": file_name,
        "num_rows": len(sub_df),
        "avg_row_negative_constraint_sum": avg_negative_violation
    })

for row in results:
    print(f"{row['file']}:{row['avg_row_negative_constraint_sum']}")
    # print(f"  average row negative constraint sum = ")
    print()

#######################
#                    file  num_rows  mean_obj_1  mean_obj_2
# 0  Mazda_RS_seed331.csv      1500    2.933758   -2.128000
# 1  Mazda_RS_seed332.csv      1500    2.931930   -2.175333
# 2  Mazda_RS_seed333.csv      1500    2.934408   -2.172667

# Mazda_RS_seed331.csv:-4.43868543076

# Mazda_RS_seed332.csv:-4.469856571266667

# Mazda_RS_seed333.csv:-4.435156679886667