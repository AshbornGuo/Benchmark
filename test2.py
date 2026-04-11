import shutil
from pathlib import Path

########### 改表头
# src = Path(r"results/mazda/SBO_EGBO/Mazda_EGBO_seed333.csv")
# dst = src.with_name("Mazda_EGBO_seed333_renamed.csv")

# # 先复制
# shutil.copy2(src, dst)

# # 只修改第一行表头
# text = dst.read_text(encoding="utf-8")
# lines = text.splitlines()

# if lines:
#     lines[0] = lines[0].replace("objectives_original", "objectives", 1)
#     dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
#     print("Header renamed in:", dst)
# else:
#     print("File is empty.")

##删行数
import pandas as pd

file_path = r"results/TurbofanArch/NSGA2/TurbofanArch_NSGA2_seed331.csv"

# 关键：指定分隔符是 ;
df = pd.read_csv(file_path, sep=';')

# 只保留前 501 行
df = df.iloc[:500]

# 保存时也保持 ;
df.to_csv(file_path, sep=';', index=False)

print(df.head())
print("处理完成")


file_path2 = r"results/TurbofanArch/NSGA2/TurbofanArch_NSGA2_gentime331.csv"

# 关键：指定分隔符是 ;
df2 = pd.read_csv(file_path2, sep=';')

# 只保留前 501 行
df2 = df2.iloc[:10]

# 保存时也保持 ;
df2.to_csv(file_path2, sep=';', index=False)

print(df2.head())
print("处理完成")