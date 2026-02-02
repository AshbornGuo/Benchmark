import re

path = "dockerCall.py"
s = open(path, "r", encoding="utf-8", errors="ignore").read()

pattern = r"elif problem_name == 'HeatExchanger':.*?sys\.exit\(res\)"
replacement = """elif problem_name == 'HeatExchanger':
        import os, sys
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/HeatExchanger/heat_exchange/',
            'case_path': '/tmp/hx_case/'
        }
        os.makedirs(settings['case_path'], exist_ok=True)

        import Exeter_CFD_Problems as TestProblems
        prob = TestProblems.HeatExchanger(settings)

        x = np.fromstring(allArgs[-1], sep=',')

        # 1) 先判断“约束/几何”可行性：这才是真 infeasible
        try:
            ok = prob.constraint(x)
        except Exception as e:
            print("error")
            print("constraint_exception:", repr(e))
            sys.exit(1)

        if not ok:
            print("infeasible")
            sys.exit(2)

        # 2) 约束可行才跑 CFD：这一步失败就是 error，不是 infeasible
        try:
            res = prob.evaluate(x, verbose=verbose)
        except Exception as e:
            print("error")
            print("evaluate_exception:", repr(e))
            sys.exit(1)

        if res is None:
            print("error")
            print("evaluate_returned_none")
            sys.exit(1)

        # 3) 成功：输出两行目标值
        print(res[0])
        print(res[1])
        sys.exit(0)"""



ns, n = re.subn(pattern, replacement, s, flags=re.S)
if n != 1:
    raise RuntimeError("Failed to patch HeatExchanger block in dockerCall.py (pattern not found or multiple matches).")

open(path, "w", encoding="utf-8").write(ns)
print("patched", path)
