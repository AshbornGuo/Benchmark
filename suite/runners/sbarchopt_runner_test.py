import numpy as np
from sb_arch_opt.problems.turbofan_arch import RealisticTurbofanArch


problem = RealisticTurbofanArch()

N = 30
results = []

for i in range(N):
    # 随机生成一个候选解（1D）
    x = problem.xl + (problem.xu - problem.xl) * np.random.rand(problem.n_var)

    # 关键：correct_x 需要 2D 输入
    X = x[None, :]                       # shape: (1, n_var)

    X_corr, is_active = problem.correct_x(X)  # 都是按 batch 返回

    F = problem.evaluate(X_corr, return_values_of=["F"])

    print(i)

    results.append(F)

print(results)






















