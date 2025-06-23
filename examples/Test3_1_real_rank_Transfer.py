# author:"flt"
# data:12/24/2024 1:54 PM
import csv

# 读取原始 CSV 文件
input_filename = 'real_rank.csv'  # 请修改为你的输入文件名
output_filename = 'real_rank_care.csv'  # 输出的文件名

# 读取 CSV 数据
with open(input_filename, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)

# 定义新的参数组
new_params = ['map', 'traffic_density', 'side_detector_distance']

# 修改每三行数据的 param 列
for i, row in enumerate(rows):
    # 计算应该属于哪一组（每组3个 param）
    group_index = i % 3
    row['param'] = new_params[group_index]

# 获取 CSV 文件的列名（保留原列名）
fieldnames = reader.fieldnames

# 写入修改后的 CSV 文件
with open(output_filename, mode='w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"File has been modified and saved as {output_filename}")
