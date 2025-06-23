# author:"flt"
# data:12/1/2024 10:14 PM

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('output_verify_filter_new3.csv')

# 初始化统计结果
crash_vehicle_false_counts = []
count1_total = 0  # 统计总的正确计数
count2_total = 0  # 统计总的错误计数

# 统计每 30 条数据，每 10 条数据统计一次 'crash_vehicle' 列的 'false' 数量
for i in range(0, len(df), 30):
    # 提取当前 30 条数据
    chunk = df.iloc[i:i + 30]

    # 存储当前 30 条数据的三个统计值
    false_counts = []

    # 每 10 条数据统计一次
    for j in range(0, 30, 10):
        sub_chunk = chunk.iloc[j:j + 10]

        # 统计 'crash_vehicle' 列中 'false' 的数量
        false_count = (sub_chunk['crash_vehicle'] == False).sum()
        false_counts.append(false_count)

    # 比较这3个统计值并计算正确计数和错误计数
    count1 = 0  # 正确计数
    count2 = 0  # 错误计数

    # 判断是否为正确数据
    if false_counts[0] > false_counts[1] and false_counts[0] > false_counts[2]:
        count1 += 1
    elif false_counts[0] < false_counts[1] or false_counts[0] < false_counts[2]:
        count2 += 1
    elif (false_counts[0] == false_counts[1]) or (false_counts[0] == false_counts[2]):
        # 如果第1条数据的 false 数量与第2条或第3条数据相等，则不计入 count2
        pass
    else:
        # 如果没有符合的情况，则不改变 count1 和 count2
        pass

    # 更新总计数
    count1_total += count1
    count2_total += count2

    # 将当前统计值添加到统计列表
    crash_vehicle_false_counts.append(false_counts)

# 计算总的正确率
if count1_total + count2_total > 0:
    acc = count1_total / (count1_total + count2_total)
else:
    acc = 0.0  # 如果没有有效数据，则返回 0.0

# 将统计结果转换为 DataFrame
result_df = pd.DataFrame(crash_vehicle_false_counts, columns=['count1', 'count2', 'count3'])
result_df['accuracy'] = acc  # 添加总的正确率

# 输出到新的 CSV 文件
result_df.to_csv('false_count.csv', index=False)

print(f"统计结果及总正确率已保存到 'false_count.csv'，总正确率：{acc}")

# 如果需要输出到文件，可以使用以下代码：
# result_df = pd.DataFrame(crash_vehicle_false_counts, columns=['false_count'])
# result_df.to_csv('crash_vehicle_false_counts.csv', index=False)
