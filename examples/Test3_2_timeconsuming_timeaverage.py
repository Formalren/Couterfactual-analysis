# author:"flt"
# data:1/7/2025 11:39 AM
import pandas as pd
import csv
import os

# 读取 CSV 文件
df = pd.read_csv('output_param_diff_1.csv')

# 筛选出 param 列值为 'experiment_duration' 的行
experiment_duration_rows = df[df['param'] == 'experiment_duration']

# 计算 value_diff 总和和平均值
total_value_diff = experiment_duration_rows['value_diff'].sum()
average_value_diff = experiment_duration_rows['value_diff'].mean()

# 输出结果
print(f"Total sum of value_diff for experiment_duration: {total_value_diff}")
print(f"Average value_diff for experiment_duration: {average_value_diff}")

# 输出计时结果到 average_time_consuming.csv
output_file = 'average_time_consuming.csv'

# 如果文件不存在，或者文件为空，写入表头
if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
    with open(output_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['param', 'time_value'])  # 写入表头

# 写入数据
with open(output_file, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['total_value_diff_new', total_value_diff])
    writer.writerow(['average_value_diff_new', average_value_diff])
