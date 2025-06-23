# author:"flt"
# data:9/25/2024 10:33 PM
import pandas as pd

import pandas as pd

# 读取CSV文件
df = pd.read_csv('attribution_result1.5.csv')

# 定义参数列表
params = [
    'traffic_density', 'num_scenarios', 'accident_prob', 'map', 'daytime',
    'lidar_num_lasers', 'lidar_distance', 'lidar_gaussian_noise', 'lidar_dropout_prob',
    'side_detector_num_lasers', 'side_detector_distance', 'side_detector_dropout_prob',
    'lane_line_detector_num_lasers'
]

# 初始化结果列表
results = []

# 遍历每130条数据
for i in range(0, 13000, 130):
    chunk = df.iloc[i:i + 130]

    # 每10条统计一次0的数量
    cycle_results = []
    for j in range(0, 130, 10):
        sub_chunk = chunk.iloc[j:j + 10]
        zero_counts = sub_chunk['crash_vehicle'].value_counts().get(0, 0)

        # 将统计结果添加到周期结果列表
        cycle_results.append({
            'index': i // 130 + 1,
            'param': params[j // 10],
            'value': zero_counts
        })

    # 检查周期结果是否全为0
    if any(result['value'] > 0 for result in cycle_results):
        results.extend(cycle_results)

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 根据index和count列进行排序
sorted_results = results_df.sort_values(by=['index', 'value'], ascending=[True, False])

# 获取每个周期数量最大的三个参数
top_three_per_cycle = sorted_results.groupby('index').head(3)

# 输出到新的CSV文件
top_three_per_cycle.to_csv('output_1_5.csv', index=False)

print(top_three_per_cycle)
