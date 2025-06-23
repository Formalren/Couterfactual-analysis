#!/usr/bin/env python
"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import random
import cv2
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.constants import HELP_MESSAGE
import csv
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.examples.ppo_expert.numpy_expert import expert
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.component.map.base_map import BaseMap
import pyautogui
import os
import keyboard
import time
import threading
import math
import csv
import os


def filter_info(info):
       # 只保留指定的键
      keys_to_keep = ["crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road"]
      filtered_info = {key: info.get(key, None) for key in keys_to_keep}
      return filtered_info


def run_simulation(config, args):
    env = MetaDriveEnv(config)

    # 写入 config 数据到文件
    config_header = list(config.keys()) + ['all_min_distance', 'sum_distance', 'average_distance']
    config_file_exists = os.path.isfile('output_new.csv')

    final_info = None  # 用于保存最后一条记录
    crash_info = None  # 用于保存 crash 为 1 的记录

    try:
        o, _ = env.reset(seed=21)
        # print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        keypress_thread = threading.Thread(target=simulate_keypress)
        keypress_thread.start()

        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)

        # 初始化全局最小值
        all_min_distance = float('inf')
        sum_distance = 0
        vehicle_detected = False  # 标志位，标识是否遇到车子

        for i in range(1, 1300):
            o, r, tm, tc, info = env.step(expert(env.agent))
            # print(info)
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
                    print(distance)
                    # 碰撞时直接将 distance 设置为 0
                    if info.get("crash_vehicle") == 1:
                        distance = 0  # 碰撞时距离应为0

                    # 更新当前循环的最小距离
                    single_min_distance = min(single_min_distance, distance)
                    vehicle_detected = True  # 如果检测到车子，则更新标志

            # # 对 info 进行过滤
            # filtered_info = filter_info(info)
            # final_info = filtered_info  # 更新最后一条记录

            # ego与vehicle碰撞时记录相关参数信息
            if info.get("crash_vehicle") == 1:
                crash_info = filter_info(info)  # 更新 crash 为 1 的记录
                # single_min_distance = 0  # 碰撞时 single_min_distance 应为 0

            # 更新全局最小距离
            all_min_distance = min(all_min_distance, single_min_distance)

            # 只有在遇到车子时，才累加 single_min_distance
            if vehicle_detected:
                sum_distance += single_min_distance
                print(sum_distance)
                vehicle_detected = False  # 重置标志位，等待下一次检测

            # env.render(
            #     text={
            #         "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
            #         "Current Observation": args.observation,
            #         "Keyboard Control": "W,A,S,D",
            #     }
            # )
            print("Navigation information: ", info["navigation_command"])

            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)
            # if args.observation == "rgb_camera":
            #         cv2.imshow('RGB Image in Observation', o["image"][..., -1])
            #         cv2.waitKey(1)
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

        # 将结果写入 output.csv
        with open('output_new.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=config_header + list(info_to_write.keys()))
            if not config_file_exists:
                writer.writeheader()
            writer.writerow({
                **config,
                **info_to_write,
                'all_min_distance': all_min_distance,
                'sum_distance': sum_distance,
                'average_distance': average_distance
            })
        env.close()



def simulate_keypress():
    time.sleep(15)
    pyautogui.press('f')


if __name__ == "__main__":
    for _ in range(400):
        config = dict(
            # use_render=True,
            # image_observation=True,
            agent_policy=ExpertPolicy,
            # traffic_density=random.choice([0.3]),
            num_scenarios=2000,
            random_agent_model=False,
            accident_prob=0.45,
            # random_lane_width=True,
            # random_lane_num=True,
            # on_continuous_line_done=False,
            # out_of_route_done=True,
            vehicle_config=dict(show_lidar=True, show_navi_mark=True, show_line_to_navi_mark=True),
            # map=random.choice([3]),
            start_seed=10,
            use_render=True,
            # manual_control=False,
            traffic_density=0.2,
            # environment_num=100,
            # random_agent_model=True,
            random_lane_width=True,
            random_lane_num=True,
            show_skybox=True,
            map=5  # seven block
            # start_seed=random.randint(0, 1000)
        )
        parser = argparse.ArgumentParser()
        parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
        args = parser.parse_args()
        # if args.observation == "semantic_camera":
        #     config.update(
        #         dict(
        #             image_observation=True,
        #             sensors = dict(semantic_camera=(SemanticCamera, 400, 300)),
        #             # sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
        #             interface_panel=["semantic_camera", "dashboard"]
        #         )
        #     )
        if args.observation == "rgb_camera":
            config.update(
                dict(
                    image_observation=True,
                    sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
                    interface_panel=["rgb_camera", "dashboard"]
                )
            )

        config.update(dict(
            vehicle_config=dict(
                lidar=dict(num_lasers=4, distance=50,
                                   gaussian_noise=0.04,
                                   dropout_prob=0.6,
                    add_others_navi=False,
                ),
                side_detector=dict(num_lasers=2, distance=20,
                                   gaussian_noise=0.04,
                                   dropout_prob=0.08),
                lane_line_detector=dict(num_lasers=1, distance=20,
                                   gaussian_noise=0.04,
                                   dropout_prob=0.08)
            ),
            daytime=f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
        ))

        run_simulation(config, args)

