# suite/runners/mechbench_runner.py
import subprocess
from suite.utils import save_result, Timer

# 这里改成你本地MECHBench源码所在目录（相对 run_all.py）
# MECHBENCH_DIR = "third_party/MECHBench"
MECHBENCH_DIR = r"C:/Users/guoji\Desktop/graduate project/codes/benchmarks/MECHBench/MECHBench"


def run_mechbench(suite, problem, algo, seed, budget, **kwargs):
    """
    最小跑通版本（不改MECHBench main.py）：
    - 直接运行 `python main.py`
    - 从 stdout 里拿到最后一行作为 obj_value（也就是 F）
    - 写入统一 results/*.json
    注意：seed/budget 现在不会影响 main.py（因为 main.py 里是写死的），
    我们先跑通框架，后面再让它可控。
    """
    cmd = ["python", "main.py"]

    with Timer() as t:
        p = subprocess.run(cmd, cwd=MECHBENCH_DIR, capture_output=True, text=True)

    status = "ok" if p.returncode == 0 else "failed"

    # 尝试从 main.py 的输出里读出 obj_value
    F = None
    if status == "ok":
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip() != ""]
        if len(lines) > 0:
            last = lines[-1]
            # 先当成“单目标的一个数”
            # 如果它打印的是数组/列表，我们先保留字符串也没关系
            try:
                F = float(last)
            except Exception:
                F = last  # 不是纯数字就先保存原字符串

    out_path = f"results/{suite}/{problem}/{algo}/seed_{seed}.json"
    save_result(
        {
            "suite": suite,
            "problem": problem,
            "algorithm": algo,
            "seed": seed,
            "budget": budget,
            "status": status,
            "time_sec": t.elapsed,
            "F": F,
            # 调试信息：如果 failed，你打开json就能看到错误原因
            "stdout_tail": p.stdout[-2000:],
            "stderr_tail": p.stderr[-2000:],
        },
        out_path,
    )
