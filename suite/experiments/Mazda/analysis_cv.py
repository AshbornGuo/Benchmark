import pandas as pd
import numpy as np
import ast

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULT_DIR = PROJECT_ROOT / "results" / "mazda" / "random_search"

PATH_RS_331 = RESULT_DIR / "Mazda_RS_seed331.csv"
PATH_RS_332 = RESULT_DIR / "Mazda_RS_seed332.csv"
PATH_RS_333 = RESULT_DIR / "Mazda_RS_seed333.csv"

paths = [PATH_RS_331, PATH_RS_332, PATH_RS_333]

all_objectives = []
all_negative_constraint_sums = []

for path in paths:
    df = pd.read_csv(path, sep=';')
    

    obj_list = df["objectives"].apply(ast.literal_eval)
    all_objectives.extend(obj_list.tolist())

    cons_list = df["constraints"].apply(ast.literal_eval)
    for cons in cons_list:
        neg_sum = sum(x for x in cons if x < 0)
        all_negative_constraint_sums.append(neg_sum)


all_objectives = np.array(all_objectives, dtype=float)
mean_objectives = all_objectives.mean(axis=0)


mean_constraints = np.mean(all_negative_constraint_sums)

print("mean_objectives:", mean_objectives)
print("mean_constraints:", mean_constraints)