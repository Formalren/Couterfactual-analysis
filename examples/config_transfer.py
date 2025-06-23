# author:"flt"
# data:8/22/2024 10:26 PM
import csv
import json

import csv
import ast

# 打开 input 和 output 文件
with open('filtered_output.csv', mode='r', encoding='utf-8') as infile, open('output_end.csv', mode='w',
                                                                             encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)

    # 读取原始 CSV 文件中的字段名
    original_fieldnames = reader.fieldnames

    # 定义输出 CSV 的字段名
    fieldnames = original_fieldnames + [
        'lidar_num_lasers', 'lidar_distance', 'lidar_gaussian_noise', 'lidar_dropout_prob', 'lidar_add_others_navi',
        'side_detector_num_lasers', 'side_detector_distance', 'side_detector_gaussian_noise',
        'side_detector_dropout_prob',
        'lane_line_detector_num_lasers', 'lane_line_detector_distance', 'lane_line_detector_gaussian_noise',
        'lane_line_detector_dropout_prob'
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()  # 写入 CSV 文件头

    for row in reader:
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

            # 写入合并后的数据
            writer.writerow(combined_data)

        except (ValueError, KeyError) as e:
            print(f"Skipping row due to error: {e}")
            continue  # 跳过解析失败的行