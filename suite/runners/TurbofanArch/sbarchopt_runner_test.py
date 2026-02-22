import numpy as np
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch
import time


problem = RealisticTurbofanArch()

N = 20
results = []

t0 = time.perf_counter()
for i in range(N):
    # 随机生成一个候选解（1D）
    x = problem.xl + (problem.xu - problem.xl) * np.random.rand(problem.n_var)

    # 关键：correct_x 需要 2D 输入
    X = x[None, :]                       # shape: (1, n_var)

    X_corr, is_active = problem.correct_x(X)  # 都是按 batch 返回

    F = problem.evaluate(X_corr, return_values_of=["F"])

    print(i)

    results.append(F)
t1 = time.perf_counter()
print("time:",t1-t0, "results:",results)






















