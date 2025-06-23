import numpy as np, pandas as pd, networkx as nx
from dowhy import gcm
from array import array
import os
import pickle
import sys
import pandas as pd
from ananke.datasets import load_afixable_data
from ananke.identification import OneLineID
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


from ananke.graphs import ADMG
from networkx import DiGraph

import csv

# X = np.random.uniform(low=-5, high=5, size=1000)
# # print(X)
# Y = 0.5 * X + np.random.normal(loc=0, scale=1, size=1000)
# Z = 2 * Y + np.random.normal(loc=0, scale=1, size=1000)
# W = 3 * Z + np.random.normal(loc=0, scale=1, size=1000)
# data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z, W=W))

fisherz = "fisherz"
# 方法一：直接将csv数据转换为numpy特征值
# data = []
# with open('observational_data/output_end.csv', 'r') as file:
#     csv_reader = csv.reader(file)
#     headers = next(csv_reader)
#
#     for row in csv_reader:
#         data.append(row)
#
# array = np.array(data)


# def run_care(rcausal, df, tabu_edges_remove, columns, objectives, alpha, NUM_PATHS):
#     fci_edges, G = rcausal.care_fci(df, tabu_edges_remove, alpha)
#     edges = []
#     # resolve notears_edges and fci_edges and update
#     di_edges, bi_edges = rcausal.resolve_edges(G, df, edges, fci_edges, columns,
#                                                tabu_edges_remove, objectives, NUM_PATHS)
#     G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)
#     return G, di_edges, bi_edges



df = pd.read_csv('examples/output_fci.csv')
#
# 获取列名
columns = df.columns.tolist()

# 创建字典，并将每一列的数据分别存储在对应的键中
data = {col: df[col].tolist() for col in columns}
# print(data['traffic_density'])

data = pd.DataFrame(
    data=dict(traffic_density=data['traffic_density'], num_scenarios=data['num_scenarios'], accident_prob=data['accident_prob'],
              map=data['map'],
              daytime=data['daytime'], crash_vehicle=data['crash_vehicle'], crash_object=data['crash_object'],
              crash_building=data['crash_building'],
              crash_human=data['crash_human'], crash_sidewalk=data['crash_sidewalk'], out_of_road=data['out_of_road'],
              lidar_num_lasers=data['lidar_num_lasers'],
              lidar_distance=data['lidar_distance'], lidar_gaussian_noise=data['lidar_gaussian_noise'],
              lidar_dropout_prob=data['lidar_dropout_prob'], side_detector_distance=data['side_detector_distance'],
              side_detector_gaussian_noise=data['side_detector_gaussian_noise'], side_detector_dropout_prob=data['side_detector_dropout_prob'],
              lane_line_detector_distance=data['lane_line_detector_distance'], lane_line_detector_gaussian_noise=data['lane_line_detector_gaussian_noise'],
              lane_line_detector_dropout_prob=data['lane_line_detector_dropout_prob']))  # This data frame consist of only one sample here.
# print(df)
# columns = df.columns
# opt = ['traffic_density', 'num_scenarios', 'map', 'daytime',
#        'accident_prob', 'lidar_distance', 'lidar_num_lasers', 'lidar_gaussian_noise',
#        'lidar_dropout_prob',
#        'side_detector_distance', 'side_detector_gaussian_noise', 'side_detector_dropout_prob',
#        'lane_line_detector_distance', 'lane_line_detector_gaussian_noise', 'lane_line_detector_dropout_prob']
# options = df[opt]
# # met = []
# # metrics = df[met]
# obj = ['crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'crash_sidewalk', 'out_of_road']
# objectives = df[obj]
# g, edges = fci(array.astype(float), fisherz, 0.05, verbose=True)
#
# fci_edges = []
# for edge in edges:
#     fci_edges.append(str(edge))

# print(fci_edges)
# CM = rcausal(columns)
# g = DiGraph()
# g.add_nodes_from(columns)
#
# alpha = 0.8  # use 0.8
# NUM_PATHS = 1
# # edge constraints
#
# tabu_edges_remove = CM.get_tabu_edges_care_remove(options, objectives)
# G, di_edges, bi_edges = run_care(CM, df, tabu_edges_remove,
#                                  columns, objectives, alpha, NUM_PATHS)


# print(data)
# 'overtake_vehicle_num', 'velocity', 'steering', 'acceleration', 'step_energy', 'episode_energy', 'env_seed','route_completion', 'cost', 'step_reward','episode_reward', 'episode_length'
# 列出因果关系
causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph(
    [('lidar_gaussian_noise', 'crash_human'), ('map', 'crash_vehicle'),
     ('lane_line_detector_dropout_prob', 'crash_human'), ('lidar_gaussian_noise', 'crash_sidewalk'),
     ('num_scenarios', 'out_of_road'), ('lidar_gaussian_noise', 'crash_object'), ('num_scenarios', 'crash_building'),
     ('lidar_gaussian_noise', 'crash_vehicle'), ('lane_line_detector_distance', 'crash_building'),
     ('lane_line_detector_distance', 'crash_human'), ('lane_line_detector_distance', 'crash_sidewalk'),
     ('accident_prob', 'out_of_road'), ('num_scenarios', 'crash_human'), ('num_scenarios', 'crash_sidewalk'),
     ('num_scenarios', 'crash_object'), ('accident_prob', 'crash_building'),
     ('side_detector_gaussian_noise', 'crash_building'), ('accident_prob', 'crash_human'),
     ('side_detector_gaussian_noise', 'crash_human'), ('num_scenarios', 'crash_vehicle'),
     ('traffic_density', 'out_of_road'), ('side_detector_gaussian_noise', 'crash_sidewalk'),
     ('side_detector_gaussian_noise', 'crash_object'), ('daytime', 'crash_building'),
     ('traffic_density', 'crash_building'), ('accident_prob', 'crash_vehicle'), ('daytime', 'crash_sidewalk'),
     ('daytime', 'crash_human'), ('daytime', 'crash_object'), ('side_detector_dropout_prob', 'crash_building'),
     ('lane_line_detector_gaussian_noise', 'out_of_road'), ('traffic_density', 'crash_human'),
     ('traffic_density', 'crash_object'), ('lane_line_detector_gaussian_noise', 'crash_building'),
     ('map', 'out_of_road'), ('side_detector_dropout_prob', 'crash_human'), ('lidar_num_lasers', 'crash_building'),
     ('lidar_distance', 'crash_building'), ('lidar_dropout_prob', 'out_of_road'), ('traffic_density', 'crash_vehicle'),
     ('side_detector_distance', 'crash_building'), ('map', 'crash_building'),
     ('lane_line_detector_gaussian_noise', 'crash_human'), ('lidar_dropout_prob', 'crash_building'),
     ('lane_line_detector_gaussian_noise', 'crash_sidewalk'), ('lidar_num_lasers', 'crash_human'),
     ('lidar_distance', 'crash_human'), ('lidar_num_lasers', 'crash_sidewalk'), ('map', 'crash_human'),
     ('side_detector_distance', 'crash_human'), ('lane_line_detector_dropout_prob', 'out_of_road'),
     ('map', 'crash_sidewalk'), ('lidar_distance', 'crash_object'), ('lidar_dropout_prob', 'crash_human'),
     ('map', 'crash_object'), ('lidar_gaussian_noise', 'crash_building'),
     ('lane_line_detector_dropout_prob', 'crash_building'), ('lane_line_detector_distance', 'out_of_road')]
    ))  # X -> Y -> Z -> W
# #将因果关系和数据拟合
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)
# # print(gcm)


traffic_density = 3
#确认场景含义？
num_scenarios = 10000
accident_prob = 0.2
map = 10
daytime = 3
crash_vehicle = 1
crash_object = 1
crash_building = 1
crash_human = 1
crash_sidewalk = 1
out_of_road = 1
lidar_num_lasers = 225
lidar_distance = 1000
lidar_gaussian_noise = 200
lidar_dropout_prob = 5
side_detector_distance = 125
side_detector_gaussian_noise = 0.1
side_detector_dropout_prob = 0.1
lane_line_detector_distance = 125
lane_line_detector_gaussian_noise = 0.1
lane_line_detector_dropout_prob = 0.1

#
anomalous_data = pd.DataFrame(
    data=dict(traffic_density=[traffic_density], num_scenarios=[num_scenarios], accident_prob=[accident_prob],
              map=[map],
              daytime=[daytime], crash_vehicle=[crash_vehicle], crash_object=[crash_object],
              crash_building=[crash_building],
              crash_human=[crash_human], crash_sidewalk=[crash_sidewalk], out_of_road=[out_of_road],
              lidar_num_lasers=[lidar_num_lasers],
              lidar_distance=[lidar_distance], lidar_gaussian_noise=[lidar_gaussian_noise],
              lidar_dropout_prob=[lidar_dropout_prob], side_detector_distance=[side_detector_distance],
              side_detector_gaussian_noise=[side_detector_gaussian_noise], side_detector_dropout_prob=[side_detector_dropout_prob],
              lane_line_detector_distance=[lane_line_detector_distance], lane_line_detector_gaussian_noise=[lane_line_detector_gaussian_noise],
              lane_line_detector_dropout_prob=[lane_line_detector_dropout_prob]))  # This data frame consist of only one sample here.
attribution_scores = gcm.attribute_anomalies(causal_model, 'crash_object', anomaly_samples=anomalous_data)
print(attribution_scores)
# overtake_vehicle_num=0
# velocity=9.325
# steering=-0.03
# acceleration=1
# step_energy=0.004
# episode_energy=1
# env_seed=21
# step_reward=0.955
# route_completion=0.15
# cost=0
# episode_reward=100
# episode_length=150
# anomalous_data = pd.DataFrame(data=dict(overtake_vehicle_num=[overtake_vehicle_num], velocity=[velocity], steering=[steering], acceleration=[acceleration],
#                             step_energy=[step_energy],episode_energy=[episode_energy],env_seed=[env_seed],step_reward=[step_reward],
#                             route_completion=[route_completion],cost=[cost],episode_reward=[episode_reward],episode_length=[episode_length]))  # This data frame consist of only one sample here.
#
# attribution_scores = gcm.attribute_anomalies(causal_model, 'episode_energy', anomaly_samples=anomalous_data)
# attribution_scores
#     {'X': array([0.59766433]), 'Y': array([7.40955119]), 'Z': array([-0.00236857]), 'W': array([0.0018539])}
