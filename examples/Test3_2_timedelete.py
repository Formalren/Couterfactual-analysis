# author:"flt"
# data:1/7/2025 1:02 PM
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('output_param_diff.csv')

# 删除 param 列中值为 'experiment_duration' 的行
df_cleaned = df[df['param'] != 'experiment_duration']

# 输出删除后的 DataFrame
print(df_cleaned)

# 如果需要将清理后的数据保存到新的 CSV 文件
df_cleaned.to_csv('new_file.csv', index=False)
