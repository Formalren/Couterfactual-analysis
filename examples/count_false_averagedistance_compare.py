# author:"flt"
# data:12/16/2024 11:06 AM
import pandas as pd
import pandas as pd
import numpy as np
import math
# 读取 CSV 文件
df = pd.read_csv('output_verify_filter_new11.csv')

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
    average_distances = []

    # 每 10 条数据统计一次
    for j in range(0, 30, 10):
        #在尝试对 DataFrame 切片进行修改时，这会导致 Pandas 无法确定修改是否会影响原始 DataFrame，
        # 或者是否对副本进行。即使使用了.loc，如果sub_chunk是从原始 DataFrame 中剪切出来的一个视图（而不是）副本），
        # 也可能会导致这种警告。
        sub_chunk = chunk.iloc[j:j + 10].copy()  # 显式创建副本

        # 使用 .loc 修改 'average_distance' 列的值，避免 SettingWithCopyWarning
        sub_chunk.loc[:, 'all_min_distance'] = sub_chunk['all_min_distance'].apply(
            lambda x: 100 if np.isinf(x) else x
        )

        # 统计 'crash_vehicle' 列中 'false' 的数量
        false_count = (sub_chunk['crash_vehicle'] == False).sum()
        false_counts.append(false_count)

        # 计算 'average_distance' 列的平均值
        average_distance = sub_chunk['all_min_distance'].mean()
        average_distances.append(average_distance)

    # 比较这3个统计值并计算正确计数和错误计数
    count1 = 0  # 正确计数
    count2 = 0  # 错误计数

    # 判断是否为正确数据
    if false_counts[0] > false_counts[1] and false_counts[0] > false_counts[2]:
        count1 += 1
    elif false_counts[0] < false_counts[1] or false_counts[0] < false_counts[2]:
        count2 += 1
    elif false_counts[0] == false_counts[1] or false_counts[0] == false_counts[2]:
        # 如果第1条数据的 false 数量与第2条或第3条数据相等，比较 average_distance 平均值
        if false_counts[0] == false_counts[1] == false_counts[2]:
            if average_distances[0] > average_distances[1] and average_distances[0] > average_distances[2]:
                count1 += 1
            else:
                count2 += 1
        elif false_counts[0] == false_counts[1]:
            if average_distances[0] > average_distances[1]:
                count1 += 1
            else:
                count2 += 1
        elif false_counts[0] == false_counts[2]:
            if average_distances[0] > average_distances[2]:
                count1 += 1
            else:
                count2 += 1
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
    #bodonglv
    volatility = math.sqrt(acc * (1 - acc))
else:
    acc = 0.0  # 如果没有有效数据，则返回 0.0

# 将统计结果转换为 DataFrame
result_df = pd.DataFrame(crash_vehicle_false_counts, columns=['count1', 'count2', 'count3'])
result_df['accuracy'] = acc  # 添加总的正确率f
result_df['volatility'] = volatility

# 输出到新的 CSV 文件
result_df.to_csv('false_count.csv', index=False)

print(f"统计结果及总正确率已保存到 'false_count.csv'，总正确率：{acc},波动率：{volatility}")
