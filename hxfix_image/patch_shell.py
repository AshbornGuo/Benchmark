import re

path = "dockerCall.sh"
s = open(path, "r", encoding="utf-8", errors="ignore").read()

s2 = re.sub(
    r'if \[\[ \$1 == "HeatExchanger" \]\]; then.*?fi',
    'if [[ $1 == "HeatExchanger" ]]; then\n  echo "$res" | tail -n 2\nfi',
    s,
    flags=re.S
)

open(path, "w", encoding="utf-8").write(s2)
print("patched", path)
