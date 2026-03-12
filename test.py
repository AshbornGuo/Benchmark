import shutil
from pathlib import Path

src = Path(r"results/LayeredBeam/EGBO/LayeredBeam_EGBO_seed331.csv")
dst = src.with_name("LayeredBeam_EGBO_seed331_renamed.csv")

# 先复制
shutil.copy2(src, dst)

# 只修改第一行表头
text = dst.read_text(encoding="utf-8")
lines = text.splitlines()

if lines:
    lines[0] = lines[0].replace("objectives_original", "objectives", 1)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Header renamed in:", dst)
else:
    print("File is empty.")