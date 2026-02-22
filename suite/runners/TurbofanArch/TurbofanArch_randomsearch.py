import numpy as np
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch
import time
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path


seed  = 331
np.random.seed(seed)
problem = RealisticTurbofanArch()

PATH_RESULT   = r"C:/Users/guoji/Desktop/python3_11_test/results/TurbofanArch/random_search"
os.makedirs(PATH_RESULT, exist_ok=True)

N = 10


df = pd.DataFrame(
    columns=[
        "objectives",
        "is_feasible",
        "variables",
        "constraints",
        "algorithm_time",
        "evaluation_time",
    ]
)


for i in range(N):
    # 随机生成候选解
    t0 = time.perf_counter()
    x = problem.xl + (problem.xu - problem.xl) * np.random.rand(problem.n_var)
    X = x[None, :]
    
    t1 = time.perf_counter()

    # 修复
    X_corr, is_active = problem.correct_x(X)

    # 同时请求 F/G/H/CV
    out = problem.evaluate(X_corr, return_values_of=["F", "G"])
    
    t2 = time.perf_counter()
    # ---- 关键：out 可能是 tuple（你现在就是这种）----
    # 约定顺序与 return_values_of 一致
    F, G = out

    # 1) 是否评估成功（没有 NaN）
    is_valid = (not np.isnan(F).any())

    df.loc[len(df)] = {
        "objectives": F.flatten().tolist(),
        "is_feasible": is_valid,
        "variables": X_corr.flatten().tolist(),
        "constraints": G.flatten().tolist(),
        "algorithm_time": t1-t0,
        "evaluation_time": t2-t1,
    }

RESULT_DIR = Path(PATH_RESULT)   
RESULT_DIR.mkdir(parents=True, exist_ok=True)
out_file = RESULT_DIR / f"TurbofanArch_randomsearch_seed{seed}.csv"
df.to_csv(out_file, index=False, sep=";")



# ###################
# import numpy as np
# from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch
# import time
# import os
# import time
# import numpy as np
# import pandas as pd
# from pathlib import Path


# seed  = 331
# np.random.seed(seed)
# problem = RealisticTurbofanArch()


# from sb_arch_opt.algo.pymoo_interface import plot

# # Plot the Pareto front
# # Normally only test problems have access to the "real" Pareto front!
# # plot(problem.pareto_front())
# pf = problem.pareto_front()
# print(pf[:5])