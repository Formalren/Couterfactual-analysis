# author:"flt"
# data:8/28/2024 10:38 PM
import pandas as pd
import csv
import ast
from io import StringIO

# 输入和输出文件名
from pandas import DataFrame

input_file = 'output.csv'
final_output_file = 'output_end1.csv'

# 第一步：读取CSV文件并转换布尔值，将数据保存到一个字符串缓冲区
with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    # fieldnames = reader.fieldnames
    fieldnames = reader.fieldnames + ['original_daytime']  # 添加新列名
    output_buffer = StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    for row in reader:
        row['original_daytime'] = row['daytime']  # 保存原始 daytime
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

# 第三步：处理 vehicle_config 字段并写入最终输出文件
with open(final_output_file, mode='w', encoding='utf-8', newline='') as outfile:
    original_fieldnames = df.columns.tolist()
    fieldnames = original_fieldnames + [
        'lidar_num_lasers', 'lidar_distance', 'lidar_gaussian_noise', 'lidar_dropout_prob', 'lidar_add_others_navi',
        'side_detector_num_lasers', 'side_detector_distance', 'side_detector_gaussian_noise',
        'side_detector_dropout_prob',
        'lane_line_detector_num_lasers', 'lane_line_detector_distance', 'lane_line_detector_gaussian_noise',
        'lane_line_detector_dropout_prob'
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()  # 写入 CSV 文件头

    for _, row in df.iterrows():
        try:
            # 将字符串转换为字典（注意处理单引号）
            vehicle_config = ast.literal_eval(row['vehicle_config'])

            # 构建提取的字段字典
            extracted_data = {
                'lidar_num_lasers': vehicle_config['lidar']['num_lasers'],
                'lidar_distance': vehicle_config['lidar']['distance'],
                'lidar_gaussian_noise': vehicle_config['lidar']['gaussian_noise'],
                'lidar_dropout_prob': vehicle_config['lidar']['dropout_prob'],
                'lidar_add_others_navi': vehicle_config['lidar'].get('add_others_navi', None),
                'side_detector_num_lasers': vehicle_config['side_detector']['num_lasers'],
                'side_detector_distance': vehicle_config['side_detector']['distance'],
                'side_detector_gaussian_noise': vehicle_config['side_detector']['gaussian_noise'],
                'side_detector_dropout_prob': vehicle_config['side_detector']['dropout_prob'],
                'lane_line_detector_num_lasers': vehicle_config['lane_line_detector']['num_lasers'],
                'lane_line_detector_distance': vehicle_config['lane_line_detector']['distance'],
                'lane_line_detector_gaussian_noise': vehicle_config['lane_line_detector']['gaussian_noise'],
                'lane_line_detector_dropout_prob': vehicle_config['lane_line_detector']['dropout_prob'],
            }

            # 将原始数据和提取的数据合并
            combined_data = {**row, **extracted_data}
            # # 删除 vehicle_config 列
            # del combined_data['vehicle_config']
            # # 删除 agent_policy 列
            # del combined_data['agent_policy']
            # list = ['vehicle_config', 'agent_policy']
            # combined_data.drop(columns=['vehicle_config', 'agent_policy'],inplace = True)
            # 写入合并后的数据
            writer.writerow(combined_data)

        except (ValueError, KeyError) as e:
            print(f"Skipping row due to error: {e}")
            continue  # 跳过解析失败的行

# 重新读取文件以删除不需要的列
df = pd.read_csv(final_output_file)
# 删除 vehicle_config 和 agent_policy 列
df.drop(columns=['vehicle_config', 'agent_policy', 'use_render', 'start_seed', 'lidar_add_others_navi', 'random_agent_model',
                 'random_lane_width', 'random_lane_num','on_continuous_line_done',	'out_of_route_done'], inplace=True)

# 将最终数据写入文件
df.to_csv(final_output_file, index=False)
print(f"Final data has been written to {final_output_file}")

# # 输入文件名
# input_file = 'output_Del.csv'
# final_output_file = 'output_end.csv'
# # 第一步：读取CSV文件并转换布尔值，同时处理 daytime 字段
# with open(input_file, mode='r') as infile:
#     reader = csv.DictReader(infile)
#     fieldnames = reader.fieldnames
#     output_buffer = StringIO()
#     writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
#     writer.writeheader()
#
#
#     def map_daytime_to_period(daytime):
#         hour = int(daytime.split(":")[0])
#         if 6 <= hour < 12:
#             return 1
#         elif 12 <= hour < 19:
#             return 2
#         else:
#             return 3
#
#
#     for row in reader:
#         for key, value in row.items():
#             if value.lower() == 'true':
#                 row[key] = 1
#             elif value.lower() == 'false':
#                 row[key] = 0
#         row['daytime'] = map_daytime_to_period(row['daytime'])
#         writer.writerow(row)
#
# # 将缓冲区中的数据转换为 DataFrame
# output_buffer.seek(0)
# df = pd.read_csv(output_buffer)
# # 第三步：处理 vehicle_config 字段并写入最终输出文件
# with open(final_output_file, mode='w', encoding='utf-8', newline='') as outfile:
#     # 第二步：处理 vehicle_config 字段并更新 DataFrame
#     def extract_vehicle_config(row):
#         try:
#             vehicle_config = ast.literal_eval(row['vehicle_config'])
#             return pd.Series({
#                 'lidar_num_lasers': vehicle_config['lidar']['num_lasers'],
#                 'lidar_distance': vehicle_config['lidar']['distance'],
#                 'lidar_gaussian_noise': vehicle_config['lidar']['gaussian_noise'],
#                 'lidar_dropout_prob': vehicle_config['lidar']['dropout_prob'],
#                 'lidar_add_others_navi': vehicle_config['lidar'].get('add_others_navi', None),
#                 'side_detector_num_lasers': vehicle_config['side_detector']['num_lasers'],
#                 'side_detector_distance': vehicle_config['side_detector']['distance'],
#                 'side_detector_gaussian_noise': vehicle_config['side_detector']['gaussian_noise'],
#                 'side_detector_dropout_prob': vehicle_config['side_detector']['dropout_prob'],
#                 'lane_line_detector_num_lasers': vehicle_config['lane_line_detector']['num_lasers'],
#                 'lane_line_detector_distance': vehicle_config['lane_line_detector']['distance'],
#                 'lane_line_detector_gaussian_noise': vehicle_config['lane_line_detector']['gaussian_noise'],
#                 'lane_line_detector_dropout_prob': vehicle_config['lane_line_detector']['dropout_prob']
#             })
#         except (ValueError, KeyError):
#             return pd.Series([None] * 13)
#
# df = df.join(df.apply(extract_vehicle_config, axis=1))
# df.drop(columns=['vehicle_config', 'agent_policy', 'use_render', 'start_seed', 'lidar_add_others_navi',
#                  'random_agent_model'], inplace=True)
#
# # 将最终数据写入文件
# df.to_csv(final_output_file, index=False)
# print(f"Final data has been written to {final_output_file}")