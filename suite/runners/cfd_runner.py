import subprocess
import numpy as np
from suite.utils import save_result, Timer

# 28维逐维边界（来自你打印的 lb/ub）
LB = np.array([
    -1, -1, -1, -1,          # num-pipes cheb coeffs (4)
     0,  0,  0,  0,  0,  0,   # row1: alpha(2), beta(2), omega(2)
    -1, -1,                  # row1: radii cheb coeffs(2)
     0,  0,  0,  0,  0,  0,   # row2: alpha(2), beta(2), omega(2)
    -1, -1,                  # row2: radii(2)
     0,  0,  0,  0,  0,  0,   # row3: alpha(2), beta(2), omega(2)
    -1, -1                   # row3: radii(2)
], dtype=float)

UB = np.array([
     1,  1,  1,  1,          # num-pipes cheb coeffs (4)
    10, 10, 10, 10,  1,  1,  # row1: alpha(2), beta(2), omega(2)
     1,  1,                  # row1: radii(2)
    10, 10, 10, 10,  1,  1,  # row2
     1,  1,                  # row2 radii
    10, 10, 10, 10,  1,  1,  # row3
     1,  1                   # row3 radii
], dtype=float)

DIM = 28
assert LB.shape == (DIM,) and UB.shape == (DIM,)

def run_cfd(suite, problem, algo, seed, budget, **kwargs):
    if problem != "HeatExchanger":
        raise ValueError("目前 cfd_runner 只支持 HeatExchanger")

    IMAGE = "frehbach/cfd-test-problem-suite"
    SCRIPT = "./dockerCall.sh"

    rng = np.random.default_rng(seed)

    success_outputs = []
    n_ok = 0
    n_fail = 0
    last_stdout = ""
    last_stderr = ""

    with Timer() as t:
        for i in range(budget):
            # 1) 在逐维边界内采样（关键修改点）
            x = rng.uniform(LB, UB)   # shape (28,)

            # 2) 转成 docker 要的字符串
            x_str = ",".join(str(v) for v in x)

            # 3) docker 命令
            cmd = ["docker", "run", "--rm", IMAGE, SCRIPT, problem, x_str]

            # 4) 调用 docker
            p = subprocess.run(cmd, capture_output=True, text=True)

            last_stdout = p.stdout
            last_stderr = p.stdout

            # ###
            # print("STDOUT RAW:\n", p.stdout)
            # print("STDERR RAW:\n", p.stdout)
            # ##


            if p.returncode == 0:
                n_ok += 1
                success_outputs.append({
                    "i": i,
                    "x": x.tolist(),
                    "stdout": p.stdout
                })
            else:
                n_fail += 1

    out_path = f"results/{suite}/{problem}/{algo}/seed_{seed}.json"
    save_result({
        "suite": suite,
        "problem": problem,
        "algorithm": algo,
        "seed": seed,
        "budget": budget,
        "status": "ok" if n_ok > 0 else "failed",
        "time_sec": t.elapsed,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "success_outputs": success_outputs,
        "last_stdout_tail": last_stdout[-2000:],
        "last_stderr_tail": last_stderr[-2000:],
    }, out_path)
