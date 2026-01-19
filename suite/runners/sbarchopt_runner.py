
      
      
from suite.utils import save_result, Timer

def run_sbarchopt(suite, problem, algo, seed, budget, **kwargs):
    # 这里先做最简单的：直接用你之前已经跑通的那段代码
    # 你后面再把 algo 做成可选
    from pymoo.optimize import minimize
    from sb_arch_opt.problems.hierarchical import HierZDT1Small
    from sb_arch_opt.algo.arch_sbo.api import get_arch_sbo_gp

    if problem != "HierZDT1Small":
        raise ValueError("目前示例只写了 HierZDT1Small，你可以照着加更多问题")

    prob = HierZDT1Small()
    n_init = prob.n_var * 10
    n_infill = max(0, budget - n_init)

    sbo = get_arch_sbo_gp(prob, n_parallel=4, init_size=n_init)

    with Timer() as t:
        res = minimize(prob, sbo, termination=("n_eval", n_init + n_infill), seed=seed)

    F = res.opt.get("F").tolist()

    out_path = f"results/{suite}/{problem}/{algo}/seed_{seed}.json"
    save_result({
        "suite": suite,
        "problem": problem,
        "algorithm": algo,
        "seed": seed,
        "budget": n_init + n_infill,
        "status": "ok",
        "time_sec": t.elapsed,
        "F": F
    }, out_path)
      