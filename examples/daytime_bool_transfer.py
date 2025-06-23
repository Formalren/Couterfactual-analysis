# author:"flt"
# data:8/28/2024 5:22 PM
import pandas as pd
import csv
from io import StringIO

# 输入和输出文件名
input_file = 'output.csv'
final_output_file = 'filtered_output.csv'

# 第一步：读取CSV文件并转换布尔值，将数据保存到一个字符串缓冲区
with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    output_buffer = StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    for row in reader:
        # 遍历每一列，检查是否为布尔值并转换
        for key, value in row.items():
            # 判断是否为布尔值字符串
            if value.lower() == 'true':
                row[key] = 1
            elif value.lower() == 'false':
                row[key] = 0
        # 写入转换后的行到缓冲区
        writer.writerow(row)

# 将缓冲区中的数据转换为 DataFrame
output_buffer.seek(0)
df = pd.read_csv(output_buffer)

# 第二步：根据 daytime 映射到时间段
def map_daytime_to_period(daytime):
    # 提取小时部分
    hour = int(daytime.split(":")[0])
    if 6 <= hour < 12:
        return 1
    elif 12 <= hour < 19:
        return 2
    else:
        return 3

# 应用映射函数
df['daytime'] = df['daytime'].apply(map_daytime_to_period)

# 保存结果到最终文件
df.to_csv(final_output_file, index=False)
print(f"Final data has been written to {final_output_file}")
