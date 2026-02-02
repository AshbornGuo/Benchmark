import re
import subprocess
import numpy as np
from pathlib import Path


# 28维逐维边界（按你打印的 lb/ub 对应的规则）
LB = np.array([
    -1, -1, -1, -1,          # 0-3: num-pipes Chebyshev coeffs
     0,  0,  0,  0,  0,  0,   # row1: alpha(2), beta(2), omega(2)
    -1, -1,                  # row1: radii Chebyshev coeffs(2)
     0,  0,  0,  0,  0,  0,   # row2
    -1, -1,                  # row2 radii
     0,  0,  0,  0,  0,  0,   # row3
    -1, -1                   # row3 radii
], dtype=float)

UB = np.array([
     1,  1,  1,  1,          # 0-3: num-pipes Chebyshev coeffs
    10, 10, 10, 10,  1,  1,  # row1: alpha(2), beta(2), omega(2)
     1,  1,                  # row1 radii
    10, 10, 10, 10,  1,  1,  # row2
     1,  1,                  # row2 radii
    10, 10, 10, 10,  1,  1,  # row3
     1,  1                   # row3 radii
], dtype=float)

DIM = 28
assert LB.shape == (DIM,) and UB.shape == (DIM,)


# def looks_infeasible_text(s: str) -> bool:
#     """Heuristic: treat common failure strings as infeasible."""
#     if not s:
#         return True
#     s_low = s.lower()
#     keywords = [
#         "infeasible", "constraint", "violat", "invalid", "failed",
#         "error", "exception", "traceback", "segmentation", "abort"
#     ]
#     return any(k in s_low for k in keywords)


def parse_two_floats(stdout: str):
    """
    Try to parse two objective values (T, p) from stdout.
    If cannot parse, return None.
    """
    # pick up numbers like 12.34, -0.5, 1e-3 etc.
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stdout)
    if len(nums) < 2:
        return None
    # Heuristic: many outputs include other numbers; we take the last two.
    # If your stdout format is known, we can make this stricter.
    t = float(nums[-2])
    p = float(nums[-1])
    return t, p


def main():
    # ====== user-configurable ======
    IMAGE = "cfd-suite-hxfix:v5"
    SCRIPT = "./dockerCall.sh"
    PROBLEM = "HeatExchanger"

    SEED = 0
    BUDGET = 50  # 运行次数
    OUT_TXT = Path("random_search_results.txt")
    # =================================

    rng = np.random.default_rng(SEED)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"problem={PROBLEM}\n")
        f.write(f"image={IMAGE}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"budget={BUDGET}\n")
        f.write("format: i\tstatus\tobj1\tobj2\tx(28)\n")
        f.write("-" * 120 + "\n")

        for i in range(BUDGET):
            # 1) sample within bounds
            x = rng.uniform(LB, UB)

            # 2) docker expects a comma-separated string
            x_str = ",".join(f"{v:.12g}" for v in x)

            # 3) run docker
            cmd = ["docker", "run", "--rm", IMAGE, SCRIPT, PROBLEM, x_str]
            p = subprocess.run(cmd, capture_output=True, text=True)

            stdout = p.stdout or ""
            stderr = p.stderr or ""
            ####
            print(stdout)
            print(stderr)
            ###
            status = "ok"
            obj1 = ""
            obj2 = ""

            if p.returncode != 0:
                status = "infeasible"
            else:
                # if stdout/stderr contains failure hints, mark infeasible
                if looks_infeasible_text(stdout) or looks_infeasible_text(stderr):
                    status = "infeasible"
                else:
                    parsed = parse_two_floats(stdout)
                    if parsed is None:
                        status = "infeasible"
                    else:
                        obj1, obj2 = parsed

            if status == "infeasible":
                f.write(f"{i}\tinfeasible\t\t\t{x_str}\n")
            else:
                f.write(f"{i}\tok\t{obj1}\t{obj2}\t{x_str}\n")

            # optional: print progress to console
            print(f"[{i+1}/{BUDGET}] {status}")

    print(f"\nSaved results to: {OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()
