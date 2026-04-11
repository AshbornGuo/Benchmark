import pandas as pd

file_path = r"results/TurbofanArch/SBO_qLogNEHVI/TurbofanArch_qLogNEHVI_seed332.csv"
file_path = r"results/TurbofanArch/SBO_qLogNEHVI/TurbofanArch_qLogNEHVI_seed332.csv"

# 读取 ; 分隔的 csv
df = pd.read_csv(file_path, sep=';')

# 只保留前 500 条评估数据
df = df.iloc[:500]

# 覆盖保存
df.to_csv(file_path, sep=';', index=False)

print(df.head())
print("处理完成：已保留表头 + 500条评估数据")