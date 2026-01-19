import subprocess
import numpy as np
from suite.utils import save_result, Timer

def run_cfd(suite, problem, algo, seed, budget, **kwargs):
    # 你要的：只跑 HeatExchanger
    if problem != "HeatExchanger":
        raise ValueError("目前 cfd_runner 只支持 HeatExchanger")

    IMAGE = "frehbach/cfd-test-problem-suite"
    SCRIPT = "./dockerCall.sh"
    DIM = 28

    LOW = -2.0
    HIGH = 10.0

    rng = np.random.default_rng(seed)

    success_outputs = []
    n_ok = 0
    n_fail = 0
    last_stdout = ""
    last_stderr = ""

    with Timer() as t:
        for i in range(budget):
            # 1) 随机生成 x
            x = rng.uniform(LOW, HIGH, size=DIM)

            # 2) 转成 docker 要的字符串
            x_str = ",".join(str(v) for v in x)

            # 3) docker 命令
            cmd = [
                "docker", "run", "--rm",
                IMAGE,
                SCRIPT,
                problem,
                x_str
            ]

            # 4) 调用 docker
            p = subprocess.run(cmd, capture_output=True, text=True)

            last_stdout = p.stdout
            last_stderr = p.stderr

            if p.returncode == 0:
                n_ok += 1
                success_outputs.append({
                    "i": i,
                    "x": x.tolist(),
                    "stdout": p.stdout
                })
            else:
                n_fail += 1

    # 5) 存结果（最简单：把成功的 stdout 都存进去）
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
