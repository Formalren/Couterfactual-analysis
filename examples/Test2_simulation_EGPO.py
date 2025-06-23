# author:"flt"
# data:8/30/2024 5:09 PM
import argparse
import random
import threading
import time

import cv2
import numpy as np
from metadrive.envs.Test2_metadrive_env_EGPO import MetaDriveEnv
# from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE
import csv
from metadrive.examples.ppo_expert.numpy_expert import expert
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.examples.ppo_expert.torch_expert import torch_expert
from metadrive.examples.ppo_expert.EGPO import egpo
import pyautogui
import os
import math

def filter_info(info):
    # 只保留指定的键
    keys_to_keep = ["crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road"]
    filtered_info = {key: info.get(key, None) for key in keys_to_keep}
    return filtered_info

def simulate_keypress():
    time.sleep(15)
    pyautogui.press('f')
def run_simulation(config, args):
    global sum_distance, info, all_min_distance
    env = MetaDriveEnv(config)

    # 写入 config 数据到文件
    # config_header = list(config.keys())
    config_header = list(config.keys()) + ['all_min_distance', 'sum_distance', 'average_distance']
    config_file_exists = os.path.isfile('output_verify_filter_new11.csv')

    # final_info = None  # 用于保存最后一条记录
    crash_info = None  # 用于保存 crash 为 1 的记录

    try:
        o, _ = env.reset(seed=21)
        # 打印操作信息
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        #创建并启动线程来模拟按键
        keypress_thread = threading.Thread(target=simulate_keypress)
        keypress_thread.start()
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            # 观察结果是一个以 numpy 数组作为值的字典
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            # 观察结果是一个具有形状的 numpy 数组
            print("The observation is an numpy array with shape: ", o.shape)
            # 设置的小一点，这样收集的数据少一点
        # 初始化全局最小值
        all_min_distance = float('inf')
        sum_distance = 0
        vehicle_detected = False  # 标志位，标识是否遇到车子
        for i in range(1, 1000):
            # 这里需要加入某个策略
            o, r, tm, tc, info = env.step(egpo(env.agent))
            lidar = env.engine.get_sensor("lidar")

            all_objects = lidar.get_surrounding_objects(env.agent)

            ego_position = env.agent.position

            # 初始化当前循环的最小距离为一个很大的值
            single_min_distance = float('inf')

            if len(all_objects) > 0:
                detected_vehicles = lidar.get_surrounding_vehicles(all_objects)
                for vehicle in detected_vehicles:
                    obstacle_position = vehicle.position
                    # 计算欧几里得距离
                    distance = math.sqrt((ego_position[0] - obstacle_position[0]) ** 2 +
                                         (ego_position[1] - obstacle_position[1]) ** 2)

                    # 碰撞时直接将 distance 设置为 0
                    if info.get("crash_vehicle") == 1:
                        distance = 0  # 碰撞时距离应为0

                    # 更新当前循环的最小距离
                    single_min_distance = min(single_min_distance, distance)
                    # print(single_min_distance)
                    vehicle_detected = True  # 如果检测到车子，则更新标志
            # # 对 info 进行过滤
            # filtered_info = filter_info(info)
            # final_info = filtered_info  # 更新最后一条记录

            if info.get("crash_vehicle") == 1:
                crash_info = filter_info(info)  # 更新 crash 为 1 的记录

                # 更新全局最小距离
            all_min_distance = min(all_min_distance, single_min_distance)
            # print(all_min_distance)
            # 累加 single_min_distance
            # 只有在遇到车子时，才累加 single_min_distance
            if vehicle_detected:
                sum_distance += single_min_distance
                print(sum_distance)
                vehicle_detected = False  # 重置标志位，等待下一次检测
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Current Observation": args.observation,
                    "Keyboard Control": "W,A,S,D",
                }
            )
            print("Navigation information: ", info["navigation_command"])

            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)
            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
    finally:
        # 计算 average_distance
        average_distance = sum_distance / 1000  # 除以1000是为了得到平均值
        # 选择性写入 crash_info 或 final_info
        if crash_info is not None:
            info_to_write = crash_info
        else:
            info_to_write = filter_info(info)

        # 写入 config 和 info 数据到文件
        with open('output_verify_filter_new11.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=config_header + list(info_to_write.keys()))
            if not config_file_exists:
                writer.writeheader()  # 如果文件不存在，写入 header
                # 做一次筛选
            writer.writerow({**config,
                             **info_to_write,
                             'all_min_distance': all_min_distance,
                             'sum_distance': sum_distance,
                             'average_distance': average_distance
                             })  # 合并 config 和 info 数据后写入
        env.close()
