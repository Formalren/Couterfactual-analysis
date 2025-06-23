# author:"flt"
# data:10/30/2024 11:10 PM
import pandas as pd

import pandas as pd

# 读取CSV文件
df = pd.read_csv('attribution_positive_1dot5_new.csv')

# 定义参数列表
params = [
    'traffic_density',
    'accident_prob',
    'map',
    'lidar_distance',
    'lidar_gaussian_noise',
    'lidar_dropout_prob',
    'lane_line_detector_distance',
]

# 初始化结果列表
results = []

# 遍历每70条数据
for i in range(0, 149030, 70):
    chunk = df.iloc[i:i + 70]

    # 每10条统计一次0的数量
    cycle_results = []
    for j in range(0, 70, 10):
        sub_chunk = chunk.iloc[j:j + 10]
        zero_counts = sub_chunk['crash_vehicle'].value_counts().get(0, 0)

        # 将统计结果添加到周期结果列表
        cycle_results.append({
            'index': i // 70 + 1,
            'param': params[j // 10],
            'value': zero_counts
        })

    # 检查周期结果是否全为0
    if any(result['value'] >= 0 for result in cycle_results):
        results.extend(cycle_results)

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 根据index和value列进行排序
sorted_results = results_df.sort_values(by=['index', 'value'], ascending=[True, False])

# 获取每个周期数量最大的前三个参数
top_three_per_cycle = sorted_results.groupby('index').head(3)

# 过滤掉value值相同的周期
filtered_results = top_three_per_cycle.groupby('index').filter(lambda x: len(x['value'].unique()) > 0)

# 仅保留每个周期的排名第一参数
# top_one_per_cycle = filtered_results.groupby('index').head(1)

# 输出到新的CSV文件
filtered_results.to_csv('output_1dot5_new_N.csv', index=False)

print(filtered_results)
