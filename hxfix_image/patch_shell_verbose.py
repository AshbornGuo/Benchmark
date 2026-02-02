path = "dockerCall.sh"
s = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()

out = []
for line in s:
    out.append(line)

# 我们直接重写成一个更可控的版本（保留原功能 + 出错打印日志）
new = r"""#!/bin/bash
source /opt/openfoam4/etc/bashrc
# Force serial execution / avoid PyFoam parallel & monitoring quirks
export OMPI_MCA_plm=isolated
export OMPI_MCA_rmaps_base_oversubscribe=1
export FOAM_MPI=openmpi
export WM_NCOMPPROCS=1
export OMP_NUM_THREADS=1


res=$(python3 dockerCall.py -p $1 $2 2>&1)

# 如果 python 运行过程中又打印了 dockerCall.py（通常意味着 traceback），就把完整日志打出来
if [[ "$res" == *"Traceback"* ]] || [[ "$res" == *"Exception"* ]]; then
  echo "===== BEGIN FULL LOG ====="
  echo "$res"
  echo "===== END FULL LOG ====="
  exit 1
fi
if [[ "$res" == *"infeasible"* ]]; then
  echo "infeasible"
  exit 2
fi
if [[ "$res" == *"evaluate_returned_none"* ]] || [[ "$res" == *"evaluate_exception"* ]] || [[ "$res" == *"constraint_exception"* ]]; then
  echo "error"
  exit 1
fi


# 正常输出：不同问题提取方式
if [[ $1 == "PitzDaily" ]]; then
  echo "$res" | grep -Eo '[+-]?[0-9]+(\.[0-9]+)?(e-[0-9]+)?' | head -n 1
fi
if [[ $1 == "KaplanDuct" ]]; then
  echo "$res" | grep -Eo '[+-]?[0-9]+(\.[0-9]+)?(e-[0-9]+)?' | tail -n 1
fi
if [[ $1 == "HeatExchanger" ]]; then
  # 你已 patch 过 dockerCall.py：会 print 两行目标值
  echo "$res" | tail -n 2
fi
if [[ $1 == "ESP" ]]; then
  echo "$res" | grep -Eo '[+-]?[0-9]+(\.[0-9]+)?(e-[0-9]+)?'
fi
"""
open(path, "w", encoding="utf-8").write(new + "\n")
print("patched verbose", path)
