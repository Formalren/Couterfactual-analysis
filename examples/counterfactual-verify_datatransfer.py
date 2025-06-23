# author:"flt"
# data:11/21/2024 8:31 PM
import pandas as pd

import pandas as pd

# 读取 1.csv 文件并提取第一列数据
df1 = pd.read_csv('Test2_output_1dot5_new_1.csv')
index_values = df1['index'].values

# 读取 2.csv 文件
df2 = pd.read_csv('output_end.csv')

# 初始状态，将 crash_vehicle 列设置为 0
df2['crash_vehicle'] = 0

# 遍历 1.csv 中的数字，每三个数据是同一个数字
for index_value in index_values:
    n = index_value  # 假设 index_value + 1 = n

    # 确保 n 的值在 2.csv 的行数范围内
    if 1 <= n <= len(df2):
        # 将第 n 行 crash_vehicle 的值设为 1
        df2.loc[n - 1, 'crash_vehicle'] = 1
# 保存修改后的 2.csv 文件
df2.to_csv('output_start.csv', index=False)

print("文件已成功修改并保存为 'output_start.csv'")
