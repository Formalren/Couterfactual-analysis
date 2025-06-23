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
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
import csv
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.examples.ppo_expert.numpy_expert import expert
from metadrive.component.map.base_map import BaseMap
import matplotlib._api.deprecation as mplDeprecation
import os


def filter_info(info):
    # 只保留指定的键
    keys_to_keep = ["crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road"]
    filtered_info = {key: info.get(key, None) for key in keys_to_keep}
    return filtered_info


def run_simulation(config, args):
    env = MetaDriveEnv(config)

    # 写入 config 数据到文件
    config_header = list(config.keys())
    config_file_exists = os.path.isfile('output_Del.csv')

    final_info = None  # 用于保存最后一条记录
    crash_info = None  # 用于保存 crash 为 1 的记录

    try:
        o, _ = env.reset(seed=21)
        # 打印操作信息
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        if args.observation == "rgb_camera":
         assert isinstance(o, dict)
            # 观察结果是一个以 numpy 数组作为值的字典
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            # 观察结果是一个具有形状的 numpy 数组
            print("The observation is an numpy array with shape: ", o.shape)
            # 设置的小一点，这样收集的数据少一点
        for i in range(1, 1000):
            # 这里需要加入某个策略
            o, r, tm, tc, info = env.step(expert(env.agent))
            #  o, r, tm, tc, info = env.step([0, 0])  #观察、奖励、终止、截断、步骤
            # print(info)
            # print(config)
            # datas = []
            # header = list(info.keys())
            # datas.append(info)
            # 对 info 进行过滤
            filtered_info = filter_info(info)
            final_info = filtered_info  # 更新最后一条记录

            if info.get("crash") == 1:
                crash_info = filtered_info  # 更新 crash 为 1 的记录
            # with open('output_Del.csv', 'a', newline='', encoding='utf-8') as f:
            #    writer = csv.DictWriter(f, fieldnames=header)
            #    if i == 1 and not config_file_exists:  # 仅在第一次写入数据时写入 header
            #       writer.writeheader()
            # 将true、false改0/1
            #    writer.writerows(datas)

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
        # 选择性写入 crash_info 或 final_info
        if crash_info is not None:
            info_to_write = crash_info
        else:
            info_to_write = final_info

        # 写入 config 和 info 数据到文件
        with open('output_Del.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=config_header + list(info_to_write.keys()))
            if not config_file_exists:
                writer.writeheader()  # 如果文件不存在，写入 header
                # 做一次筛选
            writer.writerow({**config, **info_to_write})  # 合并 config 和 info 数据后写入
        env.close()


if __name__ == "__main__":
    for _ in range(400):
        config = dict(
            # controller="steering_wheel",
            # unless
            use_render=True,
            agent_policy=ExpertPolicy,
            # manual_control=True,
            # useful,降低车辆密度0.3改为0.2，再改到0.1
            traffic_density=random.uniform(0, 0.1),
            # traffic_density=0.1,
            # somewhta useful
            #10000到5000，看看能不能降低内存消耗，以便一次性搜集更多数据,目前来看没有影响，设置5000-10000区间
            num_scenarios=random.randint(5000, 10000),
            # unless
            random_agent_model=False,
            #(0,1)之间撞车率太高，调到0.2，再改到0.1，更符合显示情况
            accident_prob=random.uniform(0, 0.1),
            # useful
            random_lane_width=True,
            # useful
            random_lane_num=True,
            # 连续线路完成，unless
            on_continuous_line_done=False,
            # 线路外完成，unless
            out_of_route_done=True,
            # 显示激光雷达，显示导航标记，显示线路至导航标记
            vehicle_config=dict(show_lidar=True, show_navi_mark=False, show_line_to_navi_mark=False),
            # debug=True,
            # debug_static_world=True,修改地图的大小
            map=random.randint(3, 7),  # seven block
            start_seed=10,
        )
        # 参数解析器
        parser = argparse.ArgumentParser()
        #

        parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
        args = parser.parse_args()
        #
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
                # ===== vehicle module config =====
                lidar=dict(
                    num_lasers=random.randint(150, 350), distance=random.randint(50, 200),
                    gaussian_noise=random.uniform(0, 1), dropout_prob=random.uniform(0, 0.2), add_others_navi=False
                    # gaussian_noise=0.0
                ),

                side_detector=dict(num_lasers=0, distance=random.randint(50, 200),
                                   gaussian_noise=random.uniform(0, 0.2), dropout_prob=random.uniform(0, 0.2)),
                lane_line_detector=dict(num_lasers=0, distance=random.randint(50, 200),
                                        gaussian_noise=random.uniform(0, 0.2), dropout_prob=random.uniform(0, 0.2))
            ),
            daytime=f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
            # Normal speed
            # NORMAL_SPEED=random.randint(100, 180)  # km/h
        ))
        run_simulation(config, args)
