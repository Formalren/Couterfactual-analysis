# author:"flt"
# data:1/2/2025 6:50 PM
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('output_param2.csv')

# 按照 'index' 列排序
df_sorted = df.sort_values(by='index', ascending=True)

# 删除重复的 'index' 值，保留最后一条
df_filtered = df_sorted.drop_duplicates(subset='index', keep='last')

# 保存过滤后的数据到新的 CSV 文件
df_filtered.to_csv('real_param.csv', index=False)

print("数据处理完成，过滤后的文件保存为 'real_param.csv'")
