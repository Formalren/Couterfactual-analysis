# author:"flt"
# data:8/14/2024 1:02 PM


import csv

# 输入和输出文件名
input_file = 'output_Del.csv'
output_file = 'transfer_output.csv'

# 读取CSV文件并转换布尔值
with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

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
        # 写入转换后的行
        writer.writerow(row)

print(f"Data has been written to {output_file}")