# author:"flt"
# data:1/7/2025 12:52 PM
import pandas as pd

# 第一段代码：读取并清理数据
# 读取 CSV 文件
df = pd.read_csv('output_param_diff_1.csv')

# 删除 param 列中值为 'experiment_duration' 的行
df_cleaned = df[df['param'] != 'experiment_duration']

# 第二段代码：读取1.csv和2.csv，并进行处理
# 使用清理后的 DataFrame df_cleaned 替代 1.csv
df1 = df_cleaned
df2 = pd.read_csv('output_verify_filter_new7_1.csv')

# 定义all_params
all_params = [
    'traffic_density', 'num_scenarios', 'accident_prob', 'map',
    'lidar_num_lasers', 'lidar_distance', 'lidar_gaussian_noise',
    'lidar_dropout_prob', 'side_detector_num_lasers', 'side_detector_distance',
    'side_detector_gaussian_noise', 'side_detector_dropout_prob',
    'lane_line_detector_num_lasers', 'lane_line_detector_distance',
    'lane_line_detector_gaussian_noise', 'lane_line_detector_dropout_prob'
]

# 为了存储最终排序结果
df1_sorted = pd.DataFrame(columns=['index', 'param', 'value_diff'])

# 每组读取的行数
group_size = 16
df2_group_size = 17  # 由于需要跳过第5行，因此2.csv每组应该是17行

# 遍历1.csv，每16行一个组
for group_num in range(0, len(df1), 16):
    group1 = df1.iloc[group_num:group_num + 16]
    index_value = group1['index'].iloc[0]

    # 计算2.csv中对应组的起始和结束索引
    start_idx_2 = group_num + group_num // 16  # 2.csv每组偏移增加
    end_idx_2 = start_idx_2 + 16

    # 读取对应的2.csv数据，忽略第5行
    group2 = df2.iloc[start_idx_2:end_idx_2 + 1].drop(df2.iloc[start_idx_2 + 4].name).reset_index(drop=True)  # 排除第5行

    # 为当前组过滤出有效的参数
    valid_params = []

    for param in all_params:
        # 找到对应的param在2.csv中的行
        param_idx = all_params.index(param)
        crash_vehicle_value = group2.iloc[param_idx]['crash_vehicle']

        # 如果crash_vehicle为False，加入valid_params，否则跳过
        if crash_vehicle_value == False:
            valid_params.append(param)

    # 筛选出1.csv中对应valid_params的参数
    group1_filtered = group1[group1['param'].isin(valid_params)]

    # 按value_diff从小到大排序，并只保留前三个
    group1_sorted_top3 = group1_filtered.sort_values(by='value_diff', ascending=True).head(3)

    # 合并到最终的df1_sorted
    df1_sorted = pd.concat([df1_sorted, group1_sorted_top3])


# 在df1_sorted中按index分组并做第二次筛选
df_filtered = pd.DataFrame(columns=['index', 'param', 'value_diff'])

# 对 DataFrame 按 'index' 分组，并筛选前两行 value_diff 是否相同
for _, group in df1_sorted.groupby('index'):
    # 第一层筛选条件：如果该组数据只有两行，则舍弃该组
    if len(group) == 2 or len(group) == 1:
        continue

    # 第二层筛选条件：如果该组数据前两行的 value_diff 相同，舍弃该组
    if len(group) > 1 and group.iloc[0]['value_diff'] == group.iloc[1]['value_diff']:
        continue

    # 如果通过筛选条件，保留该组数据
    df_filtered = pd.concat([df_filtered, group])

# 输出经过最终筛选的结果
df_filtered.to_csv('Test3_2_timedelete_rank_new2.csv', index=False)
# 输出最终筛选后的结果到 CSV 文件
# df1_sorted.to_csv('Test3_2_timedelete_rank_top3.csv', index=False)



