# run_all.py
from suite.runners.sbarchopt_runner import run_sbarchopt
from suite.runners.mechbench_runner import run_mechbench
from suite.runners.cfd_runner import run_cfd
from suite.runners.mazda_runner import run_mazda

def main():
    # 
    exp_list = [
        # {"suite": "sbarchopt", "problem": "HierZDT1Small", "algo": "SBO-GP", "seed": 0, "budget": 160},
        # {"suite": "mechbench", "problem": "crashtube_deck255", "algo": "your_algo", "seed": 0, "budget": 200},
        {"suite": "cfd", "problem": "HeatExchanger", "algo": "random_search", "seed": 2027, "budget": 1},
        # {"suite": "mazda", "problem": "CdMOBP", "algo": "random_search", "seed": 42, "budget": 200, "batch_size": 50,
        #  "excel_path": "C:/Users/guoji/Desktop/graduate project/codes/benchmarks/Mazda_Bechmark/Mazda_CdMOBP/Info_test.xlsx",
        #  "eval_exe": "C:/Users/guoji/Desktop/graduate project/codes/benchmarks/Mazda_Bechmark/Mazda_CdMOBP/Mazda_CdMOBP/bin/win64/mazda_mop.exe",
        #  "root_dir": "C:/Users/guoji/Desktop/graduate project/codes/benchmarks/Mazda_Bechmark/Mazda_CdMOBP",
        #  },
    ]

    for exp in exp_list:
        s = exp["suite"]
        if s == "sbarchopt":
            run_sbarchopt(**exp)
        elif s == "mechbench":
            run_mechbench(**exp)
        elif s == "cfd":
            run_cfd(**exp)
        elif s == "mazda":
            run_mazda(**exp)




if __name__ == "__main__":
    main()
