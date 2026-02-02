import os
import shutil
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
import csv
import subprocess
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf_discrete
from pymoo.indicators.hv import HV





num_evaluation = 5000
seed = 335
rng = np.random.default_rng(seed)
num_random_sample = 20

# Paths 
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx" # constraint file
PATH_EXE   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_DV   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/pop_vars_eval.txt"
PATH_RESULT   = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_EHVI"
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_NSGA2_seed{seed}.csv")

def dv_range(df_path) -> list:
    """
    retriet the range of dv from xlsx file

    """

    dicision_variable = pd.read_excel(df_path)
    volume_lists = []
    for _, row in dicision_variable.iterrows():
        # retriet dv and its values
        dv = row["Design Variable"]
        volume_str = row["Discrete Volume"]

        # check and skill null rows
        if pd.isna(dv) or pd.isna(volume_str):
            continue
    
        # all possible values of a dv
        values = [float(v.strip()) for v in volume_str.split(",")]
        # values of all 222 dv
        volume_lists.append(values)

    return volume_lists 

# # run the Mazda problem 
# def run_exe(exe_path, input_txt, output_dir):
#     # check and creat a new file if not exist
#     os.makedirs(output_dir, exist_ok=True) 
#     # copy input txt file into output_dir 
#     shutil.copyfile(input_txt, os.path.join(output_dir, "pop_vars_eval.txt"))
    # subprocess.run([exe_path, output_dir], check=True)
    
def run_exe(exe_path, input_txt, output_dir, timeout=60):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(input_txt, os.path.join(output_dir, "pop_vars_eval.txt"))
    subprocess.run([exe_path, output_dir], check=True, timeout=timeout)





# 
def algo_eval(path_exe, path_dv, path_result, dv_ranges) -> tuple[list[float], bool, list[float], list[float], float, float]:
    """
    retriet the range of dv from txt file
   
    :param df_path: Description
    :return: Description
    :rtype: list
    """
    
    # t_0 = time.perf_counter() # start of a random search
    
    # generate a solution usting random search
    with open(path_dv, "w") as f:

        sampled = []
        for values in dv_ranges:
            sampled.append(rng.choice(values))

        f.write("\t".join(map(str, sampled)) + "\n")
    
    # t_1 = time.perf_counter() # end of a random search; start of an evaluation

    # run exe and output to path_result
    run_exe(path_exe, path_dv, path_result)

    # t_2 = time.perf_counter() # end of an evaluation

    # algo_time = t_1 - t_0 # alforithm running time
    # eval_time = t_2 - t_1 #evaluation time

    # read 3 txt file(decision variables, ojbctives and constraints condition)
    file_dv = os.path.join(path_result, "pop_vars_eval.txt")
    file_obj = os.path.join(path_result, "pop_objs_eval.txt")
    file_con = os.path.join(path_result, "pop_cons_eval.txt")

    with open(file_obj, "r") as f:
        objs = [float(x) for x in f.read().split()]

    with open(file_dv, "r") as f:
        vars = [float(x) for x in f.read().split()]

    with open(file_con, "r") as f:
        cons = [float(x) for x in f.read().split()]

    # If all constraint values are ≥ 0, return 1 (feasible); otherwise, return 0 (infeasible)
    feasibility = all(x>=0 for x in cons)
     
    return objs, feasibility, vars, cons  

## initialization: ransom sample
def ini_sample(path_exe, path_dv, path_result, dv, num):

    cons = []
    vars = []
    objs = []

    for _ in range(num):

        # run Mazda exe
        obj_original, feasibility, var, con = algo_eval(path_exe, path_dv, path_result, dv)

        # processing symboles of the objectives and constraints
        # 因为这个算法要计算HV，所以这里就要先归一化
        f1_hv = obj_original[0] - 2.0
        f2_hv = obj_original[1] / 74.0
        obj = [f1_hv, f2_hv]

        obj_neg = [-x for x in obj] # Mazda: min; Botorch: max
        con_neg = [-x for x in con] # Mazda: >=0 is feasible; Botorch: <=0 is feasible

        cons.append(con_neg)
        objs.append(obj_neg)
        vars.append(var)
        
    return objs, cons, vars 

# 和algo_eval唯一的不同是，algo_eval是传入dv即变量的取值，然后随机生成一个候选解；而eval_given_vars是直接传入的由acq得到的一个候选解
def eval_given_vars(path_exe, path_dv, path_result, vars_one):
    # 1) 写输入（pop_vars_eval.txt）
    with open(path_dv, "w") as f:
        f.write("\t".join(map(str, vars_one)) + "\n")

    # 2) 跑 exe
    run_exe(path_exe, path_dv, path_result)

    # 3) 读输出
    file_obj = os.path.join(path_result, "pop_objs_eval.txt")
    file_con = os.path.join(path_result, "pop_cons_eval.txt")

    with open(file_obj, "r") as f:
        obj_original = [float(x) for x in f.read().split()]
    with open(file_con, "r") as f:
        con = [float(x) for x in f.read().split()]

    # 4) Mazda HV 归一化（three cars）
    f1_hv = obj_original[0] - 2.0
    f2_hv = obj_original[1] / 74.0
    obj_hv = [f1_hv, f2_hv]          # minimization

    # 5) 转 BoTorch 最大化
    obj_neg = [-x for x in obj_hv]
    con_neg = [-x for x in con]      # <=0 feasible

    return obj_neg, con_neg, obj_original, con



def compute_hv_from_history(train_Y, train_C, ref_point=(1.1, 0.0)):
    """
    train_Y: torch tensor shape (n,2), BoTorch space (max) = -obj_hv
    train_C: torch tensor shape (n,54), BoTorch constraint (<=0 feasible)
    returns: float hypervolume in Mazda HV space (min) w.r.t ref_point
    """
    # 1) 只取可行点（<=0 可行）
    feas_mask = (train_C <= 0).all(dim=1)

    if feas_mask.sum().item() == 0:
        return 0.0  # 没有可行点，HV 定义成 0 最常见

    # 2) 转回 Mazda HV 空间（min）
    Y_hv = (-train_Y[feas_mask]).detach().cpu().numpy()  # shape (k,2)

    # 3) 计算 HV（pymoo 的 HV 指标默认用于 minimization）
    hv = HV(ref_point=np.array(ref_point))
    return float(hv(Y_hv))

def norm_X(X):
    return (X - lb) / rng_


t_0 = time.perf_counter() #

# get value ranges of the decision variables
dv = dv_range(PATH_CON)


lb = torch.tensor([min(v) for v in dv], dtype=torch.double)
ub = torch.tensor([max(v) for v in dv], dtype=torch.double)
rng_ = (ub - lb).clamp_min(1e-12)



objs_neg_nor, cons_neg, vars = ini_sample(PATH_EXE, PATH_DV, PATH_RESULT, dv, num_random_sample)


# # #############
# # # 1) 初始数据（你可以随机生成 n0=20 个离散索引点）initiate data
train_X = torch.tensor(vars, dtype=torch.double)      # [n0, 222]  (索引 or 编码)
train_Y = torch.tensor(objs_neg_nor, dtype=torch.double)      # [n0, 2]    (目标：建议用 -f 做最大化)
train_C = torch.tensor(cons_neg, dtype=torch.double)      # [n0, 54]   (约束：<=0 可行)


# ref_point = [-1.1, 0.0]   # 来自文档 [1.1, 0.0] 取负
ref_point = torch.tensor([-1.1, 0.0], dtype=torch.double)


# ##现在有好几处问题被忽略了：1train_X没有归一化？？需要，也做了  
# 2.约束还没有拟合模型 3.拟合用的是默认参数吗？
# ## 4统计时间也没有做，主要是20个初始采样这里？算在1000次evaluate里面吗
# 现在的hv是是从第21次evaluate开始的
# 257次会卡死
# 可能还是要换qnlogEHVI

K = 512   # 每轮候选点数，先小一点
T = 1000 - 20      # 先跑5轮看看
hv_history = []

for it in range(T):
    # 1) fit model
    # model = SingleTaskGP(train_X, train_Y)
    train_X_n = norm_X(train_X)
    model = SingleTaskGP(train_X_n, train_Y)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 2) acq
    acq = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        # X_baseline=train_X,
        X_baseline=train_X_n,

    )

    # 3) sample candidate set
    X_cand_list = [[rng.choice(values) for values in dv] for _ in range(K)]
    X_cand = torch.tensor(X_cand_list, dtype=torch.double)
    X_cand_n = norm_X(X_cand)


    # 4) discrete optimize
    # X_next, _ = optimize_acqf_discrete(acq_function=acq, choices=X_cand, q=1)
    # 但这里有个关键：X_next_n 是归一化后的点，你还需要用它找到对应的原始点去跑 exe
    X_next_n, _ = optimize_acqf_discrete(acq_function=acq, choices=X_cand_n, q=1)

    # X_next_n 是 (1,222)，找它在 X_cand_n 的哪一行
    idx = ((X_cand_n == X_next_n).all(dim=1)).nonzero(as_tuple=True)[0].item()
    X_next = X_cand[idx:idx+1, :]   # 原始尺度 (1,222)



    # 5) true eval
    vars_next = X_next.squeeze(0).tolist()
    try:
        y_next, c_next, obj_original_next, con_next = eval_given_vars(PATH_EXE, PATH_DV, PATH_RESULT, vars_next)
    except Exception as e:
        print("eval failed:", e)
        continue

    
    # 6) append  # train_X / train_Y / train_C 里存的是到目前为止所有已经真实评估过的点
    train_X = torch.cat([train_X, X_next], dim=0) 
    train_Y = torch.cat([train_Y, torch.tensor([y_next], dtype=torch.double)], dim=0)
    train_C = torch.cat([train_C, torch.tensor([c_next], dtype=torch.double)], dim=0) 

    hv_val = compute_hv_from_history(train_Y, train_C, ref_point=(1.1, 0.0))
    hv_history.append(hv_val)

    print(f"n_eval={train_X.shape[0]}  HV={hv_val:.6f}")


t_1 = time.perf_counter() # start of a random search

print(hv_history, t_1-t_0)