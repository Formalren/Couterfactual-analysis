# author:"flt"
# data:9/7/2024 4:32 PM
import pandas as pd
import csv
from io import StringIO
import ast
import numpy as np
import networkx as nx
from dowhy import gcm
import random
import argparse
from simulation import run_simulation
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.expert_policy import ExpertPolicy
import pickle
# 第一步：读取CSV文件并处理数据
input_file = 'output-sang.csv'

with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['original_daytime']  # 添加新列名
    # fieldnames = reader.fieldnames
    output_buffer = StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
    writer.writeheader()


    def map_daytime_to_period(daytime):
        hour = int(daytime.split(":")[0])
        if 6 <= hour < 12:
            return 1
        elif 12 <= hour < 19:
            return 2
        else:
            return 3


    for row in reader:
        row['original_daytime'] = row['daytime']  # 保存原始 daytime
        for key, value in row.items():
            if value.lower() == 'true':
                row[key] = 1
            elif value.lower() == 'false':
                row[key] = 0
        row['daytime'] = map_daytime_to_period(row['daytime'])
        writer.writerow(row)

    output_buffer.seek(0)
    df = pd.read_csv(output_buffer)


    # 处理 vehicle_config 字段
    def extract_vehicle_config(row):
        try:
            vehicle_config = ast.literal_eval(row['vehicle_config'])
            return pd.Series({
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
                'lane_line_detector_dropout_prob': vehicle_config['lane_line_detector']['dropout_prob']
            })
        except (ValueError, KeyError):
            return pd.Series([None] * 12)


    df = df.join(df.apply(extract_vehicle_config, axis=1))
    df.drop(columns=['vehicle_config', 'agent_policy', 'use_render', 'start_seed', 'lidar_add_others_navi',
                     'random_agent_model',
                     'random_lane_width', 'random_lane_num', 'on_continuous_line_done', 'out_of_route_done'],
            inplace=True)
    # 第三步：因果关系建模及自动化归因
    columns = df.columns.tolist()
    data = {col: df[col].tolist() for col in columns}
    data = pd.DataFrame(data)
    causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph(
        [('accident_prob', 'crash_building'), ('lidar_dropout_prob', 'crash_sidewalk'),
         ('lane_line_detector_gaussian_noise', 'crash_sidewalk'), ('lidar_distance', 'crash_object'),
         ('accident_prob', 'crash_human'), ('lidar_dropout_prob', 'crash_building'),
         ('accident_prob', 'out_of_road'),
         ('lane_line_detector_gaussian_noise', 'crash_building'),
         ('lane_line_detector_dropout_prob', 'crash_human'),
         ('lidar_dropout_prob', 'crash_human'), ('lidar_distance', 'crash_sidewalk'),
         ('lane_line_detector_gaussian_noise', 'crash_human'), ('lidar_distance', 'crash_building'),
         ('side_detector_dropout_prob', 'crash_sidewalk'), ('lidar_distance', 'crash_human'),
         ('lane_line_detector_dropout_prob', 'crash_sidewalk'), ('side_detector_dropout_prob', 'crash_building'),
         ('lane_line_detector_dropout_prob', 'crash_building'), ('side_detector_dropout_prob', 'crash_human'),
         ('side_detector_distance', 'crash_sidewalk'), ('lidar_dropout_prob', 'out_of_road'),
         ('side_detector_num_lasers', 'crash_vehicle'), ('map', 'crash_building'),
         ('lidar_gaussian_noise', 'crash_sidewalk'), ('side_detector_distance', 'crash_building'),
         ('map', 'crash_vehicle'), ('side_detector_distance', 'crash_vehicle'), ('map', 'crash_human'),
         ('lidar_gaussian_noise', 'crash_building'), ('traffic_density', 'crash_vehicle'),
         ('side_detector_distance', 'crash_human'), ('lidar_distance', 'out_of_road'),
         ('lane_line_detector_distance', 'crash_vehicle'), ('lidar_gaussian_noise', 'crash_human'),
         ('lidar_num_lasers', 'crash_sidewalk'), ('map', 'crash_object'),
         ('side_detector_dropout_prob', 'out_of_road'),
         ('side_detector_num_lasers', 'crash_sidewalk'), ('num_scenarios', 'crash_vehicle'),
         ('lidar_num_lasers', 'crash_building'), ('map', 'crash_sidewalk'), ('num_scenarios', 'crash_human'),
         ('side_detector_gaussian_noise', 'crash_sidewalk'), ('side_detector_num_lasers', 'crash_building'),
         ('lidar_num_lasers', 'crash_human'), ('traffic_density', 'crash_sidewalk'),
         ('side_detector_gaussian_noise', 'crash_building'), ('lane_line_detector_num_lasers', 'crash_object'),
         ('side_detector_num_lasers', 'crash_human'), ('num_scenarios', 'crash_object'),
         ('lane_line_detector_distance', 'crash_sidewalk'), ('side_detector_gaussian_noise', 'crash_vehicle'),
         ('traffic_density', 'crash_building'), ('map', 'out_of_road'),
         ('side_detector_gaussian_noise', 'crash_human'),
         ('daytime', 'crash_object'), ('side_detector_distance', 'out_of_road'),
         ('lane_line_detector_distance', 'crash_building'), ('lane_line_detector_num_lasers', 'crash_sidewalk'),
         ('traffic_density', 'crash_human'), ('num_scenarios', 'crash_sidewalk'),
         ('traffic_density', 'out_of_road'),
         ('lidar_dropout_prob', 'crash_vehicle'), ('lane_line_detector_num_lasers', 'crash_building'),
         ('side_detector_gaussian_noise', 'crash_object'), ('daytime', 'crash_sidewalk'),
         ('lane_line_detector_distance', 'crash_human'), ('lane_line_detector_distance', 'out_of_road'),
         ('num_scenarios', 'crash_building'), ('accident_prob', 'crash_object'),
         ('lane_line_detector_num_lasers', 'crash_human'), ('daytime', 'crash_building'),
         ('num_scenarios', 'out_of_road'), ('daytime', 'crash_human'), ('lidar_distance', 'crash_vehicle'),
         ('accident_prob', 'crash_sidewalk'), ('daytime', 'out_of_road')]
    ))

    gcm.auto.assign_causal_mechanisms(causal_model, data)

    gcm.fit(causal_model, data)

    # 训练完模型后保存参数
    with open('causal_model.pkl', 'wb') as f:
        pickle.dump(causal_model, f)