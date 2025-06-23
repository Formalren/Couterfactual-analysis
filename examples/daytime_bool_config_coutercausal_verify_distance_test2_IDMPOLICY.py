# author:"flt"
# data:8/30/2024 5:10 PM

from Test2_simulation import run_simulation
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.expert_policy import ExpertPolicy
import pandas as pd
import random
import argparse
from metadrive.policy.idm_policy import IDMPolicy
# 读取数据
df = pd.read_csv('output_start.csv')
df_param = pd.read_csv('real_rank.csv')

# 创建一个空字典来存储结果
important_params = {}

# 遍历每一行，构建重要参数字典
for index, row in df_param.iterrows():
    idx = row['index']
    param = row['param']

    if idx not in important_params:
        important_params[idx] = []

    important_params[idx].append(param)

# 过滤出每个 index 对应的三个 param
filtered_params = {k: v for k, v in important_params.items() if len(v) == 3}
filtered_param_keys = list(filtered_params.keys())

# 初始化 params 索引f
current_param_index = 0

# 检查前100条数据并更新参数
for index, row in df.iloc[342:360].iterrows():
    if row['crash_vehicle'] == 1:
        # 提取当前行的数据
        traffic_density = row['traffic_density']
        num_scenarios = row['num_scenarios']
        accident_prob = row['accident_prob']
        map = row['map']
        original_daytime = row['original_daytime']

        lidar_params = {
            'num_lasers': row['lidar_num_lasers'],
            'distance': row['lidar_distance'],
            'gaussian_noise': row['lidar_gaussian_noise'],
            'dropout_prob': row['lidar_dropout_prob']
        }
        side_detector_params = {
            'num_lasers': row['side_detector_num_lasers'],
            'distance': row['side_detector_distance'],
            'gaussian_noise': row['side_detector_gaussian_noise'],
            'dropout_prob': row['side_detector_dropout_prob']
        }
        lane_line_detector_params = {
            'num_lasers': row['lane_line_detector_num_lasers'],
            'distance': row['lane_line_detector_distance'],
            'gaussian_noise': row['lane_line_detector_gaussian_noise'],
            'dropout_prob': row['lane_line_detector_dropout_prob']
        }

        # 获取当前 param 组
        current_params = filtered_params[filtered_param_keys[current_param_index]]
        current_param_index = (current_param_index + 1) % len(filtered_param_keys)

        # 根据当前参数组运行模拟
        for param in current_params:
            for _ in range(10):  # 每个参数进行10次模拟
                config = {
                    'use_render': True,
                    'agent_policy': IDMPolicy,
                    'random_agent_model': False,
                    'daytime': original_daytime,
                    'start_seed': 10,
                    'vehicle_config': {
                        'show_lidar': True,
                        'show_navi_mark': False,
                        'show_line_to_navi_mark': False,
                        'lidar': lidar_params.copy(),
                        'side_detector': side_detector_params.copy(),
                        'lane_line_detector': lane_line_detector_params.copy()
                    }
                }

                # Check the first parameter in current_params to determine the logic
                if current_params[0] == 'map':  # If the first parameter is 'map'
                    # Use your original logic
                    if param == 'traffic_density':
                        while True:
                            random_value = random.choice([0.08, 0.2, 0.4])
                            if random_value != traffic_density:
                                config['traffic_density'] = random_value
                                break
                    else:
                        config['traffic_density'] = traffic_density

                    if param == 'num_scenarios':
                        while True:
                            random_value = random.choice([500, 700, 900])
                            if random_value != num_scenarios:
                                config['num_scenarios'] = random_value
                                break
                    else:
                        config['num_scenarios'] = num_scenarios

                    if param == 'accident_prob':
                        while True:
                            random_value = random.choice([0.35, 0.45, 0.55])
                            if random_value != accident_prob:
                                config['accident_prob'] = random_value
                                break
                    else:
                        config['accident_prob'] = accident_prob

                    if param == 'map':
                        while True:
                            random_value = random.choice([3, 5, 7])
                            if random_value != map:
                                config['map'] = random_value
                                break
                    else:
                        config['map'] = map

                    # Randomize lidar configuration
                    if param == 'lidar_dropout_prob':
                        while True:
                            random_value = random.choice([0.5, 0.6, 0.7])
                            if random_value != lidar_params['dropout_prob']:
                                lidar_params['dropout_prob'] = random_value
                                break

                    # Randomize side_detector configuration
                    if param == 'side_detector_distance':
                        while True:
                            random_value = random.choice([10, 30, 50])
                            if random_value != side_detector_params['distance']:
                                side_detector_params['distance'] = random_value
                                break
                    if param == 'side_detector_gaussian_noise':
                        while True:
                            random_value = random.choice([0.3, 0.5, 0.7])
                            if random_value != side_detector_params['gaussian_noise']:
                                side_detector_params['gaussian_noise'] = random_value
                                break

                elif current_params[0] == 'traffic_density':  # If the first parameter is 'traffic_density'
                    # Use the alternative logic
                    if param == 'traffic_density':
                        while True:
                            random_value = random.choice([0, 0.01, 0.02])
                            if random_value != traffic_density:
                                config['traffic_density'] = random_value
                                break
                    else:
                        config['traffic_density'] = traffic_density

                    if param == 'num_scenarios':
                        while True:
                            random_value = random.choice([300, 500, 700])
                            if random_value != num_scenarios:
                                config['num_scenarios'] = random_value
                                break
                    else:
                        config['num_scenarios'] = num_scenarios

                    if param == 'accident_prob':
                        while True:
                            random_value = random.choice([0.35, 0.45, 0.55])
                            if random_value != accident_prob:
                                config['accident_prob'] = random_value
                                break
                    else:
                        config['accident_prob'] = accident_prob

                    if param == 'map':
                        while True:
                            random_value = random.choice([4, 5, 6, 7])
                            if random_value != map:
                                config['map'] = random_value
                                break
                    else:
                        config['map'] = map

                    # Randomize lidar configuration
                    if param == 'lidar_dropout_prob':
                        while True:
                            random_value = random.choice([0.4, 0.6, 0.7])
                            if random_value != lidar_params['dropout_prob']:
                                lidar_params['dropout_prob'] = random_value
                                break

                    # Randomize side_detector configuration
                    if param == 'side_detector_distance':
                        while True:
                            random_value = random.choice([0.5, 1, 2])
                            if random_value != side_detector_params['distance']:
                                side_detector_params['distance'] = random_value
                                break
                    if param == 'side_detector_gaussian_noise':
                        while True:
                            random_value = random.choice([0.1, 0.2, 0.5])
                            if random_value != side_detector_params['gaussian_noise']:
                                side_detector_params['gaussian_noise'] = random_value
                                break
                elif current_params[0] == 'lidar_dropout_prob':  # If the first parameter is 'traffic_density'
                    # Use the alternative logic
                    if param == 'traffic_density':
                        while True:
                            random_value = random.choice([0, 0.2, 0.4])
                            if random_value != traffic_density:
                                config['traffic_density'] = random_value
                                break
                    else:
                        config['traffic_density'] = traffic_density

                    if param == 'num_scenarios':
                        while True:
                            random_value = random.choice([300, 500, 700])
                            if random_value != num_scenarios:
                                config['num_scenarios'] = random_value
                                break
                    else:
                        config['num_scenarios'] = num_scenarios

                    if param == 'accident_prob':
                        while True:
                            random_value = random.choice([0.35, 0.45, 0.55])
                            if random_value != accident_prob:
                                config['accident_prob'] = random_value
                                break
                    else:
                        config['accident_prob'] = accident_prob

                    if param == 'map':
                        while True:
                            random_value = random.choice([4, 5, 6, 7])
                            if random_value != map:
                                config['map'] = random_value
                                break
                    else:
                        config['map'] = map

                    # Randomize lidar configuration
                    if param == 'lidar_dropout_prob':
                        while True:
                            random_value = random.choice([0, 0.2, 0.4])
                            if random_value != lidar_params['dropout_prob']:
                                lidar_params['dropout_prob'] = random_value
                                break

                    # Randomize side_detector configuration
                    if param == 'side_detector_distance':
                        while True:
                            random_value = random.choice([0.5, 1, 2])
                            if random_value != side_detector_params['distance']:
                                side_detector_params['distance'] = random_value
                                break
                    if param == 'side_detector_gaussian_noise':
                        while True:
                            random_value = random.choice([0.2, 0.5, 0.7])
                            if random_value != side_detector_params['gaussian_noise']:
                                side_detector_params['gaussian_noise'] = random_value
                                break
                elif current_params[0] == 'side_detector_distance':  # If the first parameter is 'traffic_density'
                    # Use the alternative logic
                    if param == 'traffic_density':
                        while True:
                            random_value = random.choice([0, 0.2, 0.4])
                            if random_value != traffic_density:
                                config['traffic_density'] = random_value
                                break
                    else:
                        config['traffic_density'] = traffic_density

                    if param == 'num_scenarios':
                        while True:
                            random_value = random.choice([300, 500, 700])
                            if random_value != num_scenarios:
                                config['num_scenarios'] = random_value
                                break
                    else:
                        config['num_scenarios'] = num_scenarios

                    if param == 'accident_prob':
                        while True:
                            random_value = random.choice([0.35, 0.45, 0.55])
                            if random_value != accident_prob:
                                config['accident_prob'] = random_value
                                break
                    else:
                        config['accident_prob'] = accident_prob

                    if param == 'map':
                        while True:
                            random_value = random.choice([4, 5, 6, 7])
                            if random_value != map:
                                config['map'] = random_value
                                break
                    else:
                        config['map'] = map

                    # Randomize lidar configuration
                    if param == 'lidar_dropout_prob':
                        while True:
                            random_value = random.choice([0, 0.2, 0.4])
                            if random_value != lidar_params['dropout_prob']:
                                lidar_params['dropout_prob'] = random_value
                                break

                    # Randomize side_detector configuration
                    if param == 'side_detector_distance':
                        while True:
                            random_value = random.choice([5, 15, 25])
                            if random_value != side_detector_params['distance']:
                                side_detector_params['distance'] = random_value
                                break
                    if param == 'side_detector_gaussian_noise':
                        while True:
                            random_value = random.choice([0.2, 0.5, 0.7])
                            if random_value != side_detector_params['gaussian_noise']:
                                side_detector_params['gaussian_noise'] = random_value
                                break
                elif current_params[0] == 'num_scenarios':  # If the first parameter is 'traffic_density'
                    # Use the alternative logic
                    if param == 'traffic_density':
                        while True:
                            random_value = random.choice([0, 0.2, 0.4])
                            if random_value != traffic_density:
                                config['traffic_density'] = random_value
                                break
                    else:
                        config['traffic_density'] = traffic_density

                    if param == 'num_scenarios':
                        while True:
                            random_value = random.choice([500, 1000, 1500])
                            if random_value != num_scenarios:
                                config['num_scenarios'] = random_value
                                break
                    else:
                        config['num_scenarios'] = num_scenarios

                    if param == 'accident_prob':
                        while True:
                            random_value = random.choice([0.35, 0.45, 0.55])
                            if random_value != accident_prob:
                                config['accident_prob'] = random_value
                                break
                    else:
                        config['accident_prob'] = accident_prob

                    if param == 'map':
                        while True:
                            random_value = random.choice([4, 5, 6, 7])
                            if random_value != map:
                                config['map'] = random_value
                                break
                    else:
                        config['map'] = map

                    # Randomize lidar configuration
                    if param == 'lidar_dropout_prob':
                        while True:
                            random_value = random.choice([0.2, 0.6, 0.7])
                            if random_value != lidar_params['dropout_prob']:
                                lidar_params['dropout_prob'] = random_value
                                break

                    # Randomize side_detector configuration
                    if param == 'side_detector_distance':
                        while True:
                            random_value = random.choice([0.5, 1, 2])
                            if random_value != side_detector_params['distance']:
                                side_detector_params['distance'] = random_value
                                break
                    if param == 'side_detector_gaussian_noise':
                        while True:
                            random_value = random.choice([0.2, 0.5, 0.7])
                            if random_value != side_detector_params['gaussian_noise']:
                                side_detector_params['gaussian_noise'] = random_value
                                break
                elif current_params[0] == 'accident_prob':  # If the first parameter is 'traffic_density'
                    # Use the alternative logic
                    if param == 'traffic_density':
                        while True:
                            random_value = random.choice([0.08, 0.2, 0.4 ])
                            if random_value != traffic_density:
                                config['traffic_density'] = random_value
                                break
                    else:
                        config['traffic_density'] = traffic_density

                    if param == 'num_scenarios':
                        while True:
                            random_value = random.choice([500, 1000, 1500])
                            if random_value != num_scenarios:
                                config['num_scenarios'] = random_value
                                break
                    else:
                        config['num_scenarios'] = num_scenarios

                    if param == 'accident_prob':
                        while True:
                            random_value = random.choice([0.05, 0.1, 0.2])
                            if random_value != accident_prob:
                                config['accident_prob'] = random_value
                                break
                    else:
                        config['accident_prob'] = accident_prob

                    if param == 'map':
                        while True:
                            random_value = random.choice([4, 5, 6, 7])
                            if random_value != map:
                                config['map'] = random_value
                                break
                    else:
                        config['map'] = map

                    # Randomize lidar configuration
                    if param == 'lidar_dropout_prob':
                        while True:
                            random_value = random.choice([0.2, 0.6, 0.7])
                            if random_value != lidar_params['dropout_prob']:
                                lidar_params['dropout_prob'] = random_value
                                break

                    # Randomize side_detector configuration
                    if param == 'side_detector_distance':
                        while True:
                            random_value = random.choice([0.5, 1, 2])
                            if random_value != side_detector_params['distance']:
                                side_detector_params['distance'] = random_value
                                break
                    if param == 'side_detector_gaussian_noise':
                        while True:
                            random_value = random.choice([0.2, 0.5, 0.7])
                            if random_value != side_detector_params['gaussian_noise']:
                                side_detector_params['gaussian_noise'] = random_value
                                break
                # Update vehicle configuration
                config['vehicle_config'].update({
                    'lidar': lidar_params.copy(),
                    'side_detector': side_detector_params.copy(),
                    'lane_line_detector': lane_line_detector_params.copy(),
                })

                # Handle observation parameters
                parser = argparse.ArgumentParser()
                parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
                args = parser.parse_args()

                if args.observation == "rgb_camera":
                    config.update({
                        'image_observation': True,
                        'sensors': {'rgb_camera': (RGBCamera, 400, 300)},
                        'interface_panel': ["rgb_camera", "dashboard"]
                    })

                # Run the simulation
                run_simulation(config, args)
