# author:"flt"
# data:10/11/2024 11:48 AM

import pandas as pd
import dowhy.gcm as gcm

# 读取 df 文件
df = pd.read_csv('output_narrow1dot5_old.csv')

# 提供的所有列的名称
columns = [
    'traffic_density',
    'num_scenarios',
    'accident_prob',
    'map',
    'daytime',
    'lidar_num_lasers',
    'lidar_distance',
    'lidar_gaussian_noise',
    'lidar_dropout_prob',
    'side_detector_num_lasers',
    'side_detector_distance',
    'side_detector_gaussian_noise',
    'side_detector_dropout_prob',
    'lane_line_detector_num_lasers',
    'lane_line_detector_distance',
    'lane_line_detector_gaussian_noise',
    'lane_line_detector_dropout_prob'
]

# 存储结果的列表
results = []

# 设定 crash_vehicle 列
crash_vehicle = df['crash_vehicle']

# 剪枝后的 for 循环，只对 crash_vehicle 和其他参数进行独立性检验
for param in columns:
    Z = crash_vehicle
    X = df[param]

    # 对列 X 和 Z 进行独立性检验
    p_value = gcm.independence_test(X, Z, method='kernel')

    # 过滤 p_value 小于 0.05 的情况,找出来的边结果是独立的
    if p_value > 0.05:
        # 将结果添加到列表中
        results.append({
            'X': param,
            'Z': 'crash_vehicle',
            'p_value': p_value
        })

# 将结果保存到 CSV 文件
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('independence_test_significant_results3.csv', index=False)
#
# # 输出完成提示
# print("Independence test results (p-value < 0.05) saved to 'independence_test_significant_results.csv'")


