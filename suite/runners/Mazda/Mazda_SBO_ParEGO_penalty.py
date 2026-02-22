import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
import csv

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf_discrete
from pymoo.indicators.hv import HV

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.sampling.normal import SobolQMCNormalSampler

# from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex


seed = 331
rng = np.random.default_rng(seed)

# 初始只生成了num_random_sample = 20个候选点，这个是要进黑盒评估的。
# 但是之后的每一轮都生成K = 512个，这个只需要在gp里筛选，然后算出最好的q=BATCH_SIZE=1个进行黑盒评估
# K = 512   
num_random_sample = 50
K = 50  #别人连续优化的迭代次数最大才200，你离散每次都选512个点，这也太大了
T = 5000 - 50  
MC_SAMPLES = 32
BATCH_SIZE = 1
PENALTY = 100.0  # 你需要调参：建议从 0.1 / 1 / 10 / 100 试起


# Paths 
PATH_CON = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/Info_test.xlsx" # constraint file
PATH_EXE   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/mazda_mop.exe"
PATH_DV   = r"C:/Users/guoji/Desktop/python3_11_test/problem_sets/mazda_interface/pop_vars_eval.txt"
PATH_RESULT   = r"C:/Users/guoji/Desktop/python3_11_test/results/mazda/SBO_ParEGO"
LOG_CSV = os.path.join(PATH_RESULT, f"Mazda_SBO_ParEGO_seed{seed}.csv")

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "objectives",
            "is_feasible",
            "variables",
            "constraints",
            "evaluation_time",
            "algorithm_time",
        ])



def dv_range(df_path) -> list:
    """
    retrieve the domain of the decision variable

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


def run_exe(exe_path, input_txt, output_dir, timeout=60):
    """
    run the Mazda .exe evaluation 

    """
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(input_txt, os.path.join(output_dir, "pop_vars_eval.txt"))
    subprocess.run([exe_path, output_dir], check=True, timeout=timeout)

def read_eval_files(path_result):
    file_dv  = os.path.join(path_result, "pop_vars_eval.txt")
    file_obj = os.path.join(path_result, "pop_objs_eval.txt")
    file_con = os.path.join(path_result, "pop_cons_eval.txt")

    with open(file_obj, "r") as f:
        objs = [float(x) for x in f.read().split()]
    with open(file_dv, "r") as f:
        dvs = [float(x) for x in f.read().split()]
    with open(file_con, "r") as f:
        cons = [float(x) for x in f.read().split()]

    return objs, dvs, cons


# 计算约束违反量
def calc_cv_from_con_raw(con_raw_list):
    """
    Mazda: con >= 0 feasible
    cv = sum(max(0, -con))  # 违反量
    feasible => cv = 0
    """
    c = torch.tensor(con_raw_list, dtype=torch.double)
    # 把原始约束变负，则原来不满足条件的约束的负值变为正，并求和，得到一个符号为正的约束违反量
    return torch.clamp(-c, min=0.0).sum().item()

def eval_vars_one(path_exe, path_dv, path_result, vars_one):
    """
    给定一个决策变量向量 vars_one:
    1) 写入 pop_vars_eval.txt
    2) 运行 exe
    3) 读取 obj / con
    4) 做你的目标变换(转 max) + 计算 cv
    返回：y_obj(max-space), cv, obj_original, con_raw, is_feasible, eval_time
    """
    # 1) write dv
    with open(path_dv, "w") as f:
        f.write("\t".join(map(str, vars_one)) + "\n")

    # 2) run exe
    t0 = time.perf_counter()
    run_exe(path_exe, path_dv, path_result)
    t1 = time.perf_counter()
    eval_time = t1 - t0

    # 3) read outputs (objs, dvs, cons)
    obj_original, _, con_raw = read_eval_files(path_result)

    # 4) transform objectives to "max" space (你原来的规则)
    f1_hv = obj_original[0] - 2.0
    f2_hv = obj_original[1] / 74.0
    y_obj = [-f1_hv, -f2_hv]

    # 5) constraint violation scalar
    cv = calc_cv_from_con_raw(con_raw)

    # 6) feasibility
    is_feasible = all(x >= 0 for x in con_raw)

    return y_obj, cv, obj_original, con_raw, is_feasible, eval_time


# 生成初始样本_运行exe_返回结果_保存txt
def ini_sample(path_exe, path_dv, path_result, dv_ranges) -> tuple[list[float], bool, list[float], list[float], float]:
    # random sample a vars vector
    sampled = [rng.choice(values) for values in dv_ranges]

    # evaluate it
    _, _, obj_original, con_raw, is_feasible, eval_time = eval_vars_one(
        path_exe, path_dv, path_result, sampled
    )

    # 这里如果你还想在 ini_sample 内写 log（可以保留）
    alg_time = None
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([obj_original, is_feasible, sampled, con_raw, eval_time, alg_time])

    # 你 pre_processing 里需要的是 obj_original / var / con_raw / eval_time
    return obj_original, is_feasible, sampled, con_raw, eval_time





# 归一化目标函数_改为max方向_构造约束违反量
# def pre_processing(path_exe, path_dv, path_result, dv, num):

def pre_processing(path_exe, path_dv, path_result, dv, num):
    cons_raw, vars_list = [], []
    objs_max, objs_pen = [], []
    cvs = []

    for _ in range(num):
        obj_original, _, dv_vals, con_raw, _ = ini_sample(path_exe, path_dv, path_result, dv)
        f1_hv = obj_original[0] - 2.0
        f2_hv = obj_original[1] / 74.0
        y_obj = [-f1_hv, -f2_hv]  # max space

        cv = calc_cv_from_con_raw(con_raw)

        # === NEW: penalty into objectives (max space => subtract) ===
        y_obj_pen = [y_obj[0] - PENALTY * cv, y_obj[1] - PENALTY * cv]

        objs_max.append(y_obj)         # 可选：保留原目标用于算HV/看真实trade-off
        objs_pen.append(y_obj_pen)     # 训练用
        cvs.append([cv])

        cons_raw.append(con_raw)
        vars_list.append(dv_vals)

    return objs_pen, objs_max, cvs, cons_raw, vars_list



def norm_X(X):
    return (X - lb) / rng_


# get value ranges of the decision variables
dv = dv_range(PATH_CON)

# d.v.的上下限
lb = torch.tensor([min(v) for v in dv], dtype=torch.double)
ub = torch.tensor([max(v) for v in dv], dtype=torch.double)
rng_ = (ub - lb).clamp_min(1e-12)

# record the first R initial samples
t_RS = time.perf_counter()

# 生成num_random_sample个初始样本
# objs_max_init, cvs_init, cons_raw_init, variables = pre_processing(PATH_EXE, PATH_DV, PATH_RESULT, dv, num_random_sample)
objs_pen_init, objs_max_init, cvs_init, cons_raw_init, variables  = pre_processing(PATH_EXE, PATH_DV, PATH_RESULT, dv, num_random_sample)




t_RE = time.perf_counter()

t_firstR = t_RE - t_RS

# # 把初始参考点的数据转成tensor形式
train_X   = torch.tensor(variables, dtype=torch.double)
train_obj = torch.tensor(objs_pen_init, dtype=torch.double)   # <-- 训练用惩罚目标
train_cv  = torch.tensor(cvs_init, dtype=torch.double)        # <-- 仅日志/分析可留


# ref_point = [-1.1, 0.0]   # 来自文档 [1.1, 0.0] 取负
ref_point = torch.tensor([-1.1, 0.0], dtype=torch.double)

    
# hv_history = []
train_X_n = norm_X(train_X)

# 2 个目标 
m_obj1 = SingleTaskGP(train_X_n, train_obj[:, 0:1])
m_obj2 = SingleTaskGP(train_X_n, train_obj[:, 1:2])

model = ModelListGP(m_obj1, m_obj2)
mll = SumMarginalLogLikelihood(model.likelihood, model)

#############################      essential BO step for qNParEGO     #########################

for i in range(T):
    t_alg0 = time.perf_counter()

    # 重新归一化（数据变了）
    train_X_n = norm_X(train_X)

    m_obj1.set_train_data(train_X_n, train_obj[:, 0].contiguous(), strict=False)
    m_obj2.set_train_data(train_X_n, train_obj[:, 1].contiguous(), strict=False)



    # 继续拟合：以上一轮超参数为起点
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # 1) 每轮采一个随机权重（M=2）
    weights = sample_simplex(2).squeeze(0)  # shape (2,)

    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)
    parego_objective = GenericMCObjective(lambda Z, X: scalarization(Z))

    best_f = parego_objective(train_obj).max().item()

    acq = qLogExpectedImprovement(
        model=model,
        objective=parego_objective,
        best_f=best_f,
        sampler=sampler,
    )

    # 后面你 optimize_acqf_discrete / eval / torch.cat 保持不变
    # --- 下面你的离散choices不变 ---
    X_cand_list = [[rng.choice(values) for values in dv] for _ in range(K)]
    X_cand = torch.tensor(X_cand_list, dtype=torch.double)
    X_cand_n = norm_X(X_cand)

    X_next_n, _ = optimize_acqf_discrete(acq_function=acq, 
                                         choices=X_cand_n, 
                                         q=BATCH_SIZE,
                                        )
    # 把X_next_n还原回原来的数值
    idx = torch.cdist(X_cand_n, X_next_n).argmin().item()
    X_next = X_cand[idx:idx+1, :]
    vars_next = X_next.squeeze(0).tolist()
    
    t_alg1 = time.perf_counter()
    alg_time = t_alg1 - t_alg0


    # exe eval
    try:
        y_obj, cv, obj_original_next, con_raw_next, is_feasible, eval_time = eval_vars_one(
            PATH_EXE, PATH_DV, PATH_RESULT, vars_next
        )

    except Exception as e:
        print("eval failed:", e)
        continue


    # is_feasible = all(x >= 0 for x in con_raw_next)

    # 更新训练集（注意：这里更新 train_obj/train_cv）
    # penalty into objectives
    y_obj_pen = [y_obj[0] - PENALTY * cv, y_obj[1] - PENALTY * cv]

    train_X = torch.cat([train_X, X_next], dim=0)
    train_obj = torch.cat([train_obj, torch.tensor([y_obj_pen], dtype=torch.double)], dim=0)

    # train_cv 你可以继续存着用于日志/统计，但不再进模型
    train_cv  = torch.cat([train_cv,  torch.tensor([[cv]], dtype=torch.double)], dim=0)
   


    # append_log_row(LOG_CSV, obj_original_next, feasibility, vars_next, con_raw_next, eval_time, alg_time)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([obj_original_next,is_feasible,vars_next,con_raw_next,eval_time,alg_time])

    print("evaluation nums:",i+1+num_random_sample)


