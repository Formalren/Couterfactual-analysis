# author:"flt"
# data:10/16/2024 8:10 PM
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from dowhy.gcm.falsify import FalsifyConst, falsify_graph, plot_local_insights, run_validations, apply_suggestions
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util import plot
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel
import os
# 创建一个简单的有向图
# g_true = nx.DiGraph()
# g_true.add_edges_from(
#  [('lane_line_detector_num_lasers', 'crash_sidewalk'), ('side_detector_dropout_prob', 'crash_vehicle'),
#   ('side_detector_num_lasers', 'crash_vehicle'), ('lane_line_detector_gaussian_noise', 'out_of_road'),
#   ('side_detector_distance', 'crash_building'), ('lidar_gaussian_noise', 'lidar_distance'),
#   ('accident_prob', 'crash_sidewalk'), ('lidar_dropout_prob', 'out_of_road'),
#   ('lidar_gaussian_noise', 'crash_building'), ('num_scenarios', 'out_of_road'), ('daytime', 'lidar_distance'),
#   ('side_detector_num_lasers', 'crash_sidewalk'), ('lane_line_detector_num_lasers', 'out_of_road'),
#   ('map', 'lidar_distance'), ('daytime', 'crash_building'), ('lidar_num_lasers', 'out_of_road'),
#   ('map', 'crash_building'), ('lidar_gaussian_noise', 'crash_human'), ('map', 'lane_line_detector_distance'),
#   ('traffic_density', 'crash_vehicle'), ('side_detector_dropout_prob', 'out_of_road'),
#   ('lane_line_detector_gaussian_noise', 'crash_human'), ('side_detector_distance', 'crash_vehicle'),
#   ('lidar_dropout_prob', 'crash_sidewalk'), ('num_scenarios', 'crash_sidewalk'),
#   ('side_detector_gaussian_noise', 'crash_human'), ('lidar_num_lasers', 'crash_sidewalk'),
#   ('accident_prob', 'crash_human'), ('side_detector_dropout_prob', 'crash_sidewalk'),
#   ('traffic_density', 'out_of_road'), ('side_detector_num_lasers', 'crash_human'),
#   ('lane_line_detector_gaussian_noise', 'crash_building'), ('side_detector_gaussian_noise', 'lidar_distance'),
#   ('lane_line_detector_gaussian_noise', 'lane_line_detector_distance'), ('side_detector_distance', 'out_of_road'),
#   ('lane_line_detector_num_lasers', 'lidar_distance'), ('side_detector_gaussian_noise', 'crash_building'),
#   ('lidar_gaussian_noise', 'crash_object'), ('lane_line_detector_num_lasers', 'crash_building'),
#   ('lane_line_detector_num_lasers', 'lane_line_detector_distance'),
#   ('lane_line_detector_dropout_prob', 'crash_sidewalk'), ('daytime', 'out_of_road'),
#   ('accident_prob', 'crash_building'), ('lidar_dropout_prob', 'crash_human'), ('traffic_density', 'crash_sidewalk'),
#   ('side_detector_num_lasers', 'lidar_distance'), ('num_scenarios', 'crash_human'), ('map', 'out_of_road'),
#   ('map', 'crash_object'), ('lidar_gaussian_noise', 'crash_vehicle'), ('side_detector_dropout_prob', 'crash_building'),
#   ('lane_line_detector_num_lasers', 'crash_human'), ('side_detector_dropout_prob', 'lane_line_detector_distance'),
#   ('side_detector_num_lasers', 'crash_building'), ('side_detector_num_lasers', 'lane_line_detector_distance'),
#   ('side_detector_distance', 'crash_sidewalk'), ('lidar_num_lasers', 'crash_human'),
#   ('lidar_gaussian_noise', 'crash_sidewalk'), ('map', 'crash_vehicle'), ('side_detector_dropout_prob', 'crash_human'),
#   ('daytime', 'crash_sidewalk'), ('map', 'crash_sidewalk'), ('lidar_dropout_prob', 'crash_building'),
#   ('num_scenarios', 'crash_building'), ('traffic_density', 'crash_building'), ('lidar_gaussian_noise', 'out_of_road'),
#   ('traffic_density', 'lane_line_detector_distance'), ('lidar_num_lasers', 'crash_building'),
#   ('lidar_num_lasers', 'lane_line_detector_distance'), ('lane_line_detector_gaussian_noise', 'crash_object'),
#   ('lane_line_detector_dropout_prob', 'crash_human'), ('traffic_density', 'crash_human'),
#   ('num_scenarios', 'crash_object'), ('side_detector_distance', 'crash_human'),
#   ('lane_line_detector_num_lasers', 'crash_object'), ('lidar_num_lasers', 'crash_object'),
#   ('accident_prob', 'crash_object'), ('lidar_dropout_prob', 'crash_vehicle'), ('num_scenarios', 'crash_vehicle'),
#   ('lane_line_detector_dropout_prob', 'lidar_distance'), ('lane_line_detector_gaussian_noise', 'crash_sidewalk'),
#   ('side_detector_dropout_prob', 'crash_object'), ('side_detector_num_lasers', 'crash_object'),
#   ('daytime', 'crash_human'), ('lane_line_detector_dropout_prob', 'crash_building'),
#   ('side_detector_gaussian_noise', 'crash_sidewalk'), ('accident_prob', 'crash_vehicle'), ('map', 'crash_human')]
#
# )
# # # # #
# # # # # # 保存为 GML 文件
# nx.write_gml(g_true, "GML_13270.gml")

# 设置随机种子，保证实验可重复
from dowhy.gcm.util.general import set_random_seed

set_random_seed(0)

# 读取DAG和数据
g_true = nx.read_gml("GML_13270.gml")
data = pd.read_csv("output_narrow1dot5_new.csv")
# # 检查图的节点是否在数据的列中
# graph_nodes = set(g_true.nodes())
# data_columns = set(data.columns)
#
# print("Graph nodes:", graph_nodes)
# print("Data columns:", data_columns)
#
# # 检查是否有不匹配
# if not graph_nodes.issubset(data_columns):
#     print("Error: Some graph nodes are not present in the data columns!")
#     print("Missing nodes:", graph_nodes - data_columns)
# print("Checking for missing values in data...")
# print(data.isnull().sum())
# # 检查是否有NaN值
# if np.any(np.isnan(data)):
#     print("数据中存在NaN值")
# else:
#     print("数据中不存在NaN值")
#
# # 打印数组内容（如果数组很大，可能需要使用部分数据）
# print(data)

# 显示真实的DAG
print("True DAG")
plot(g_true)

# 输出文件名
output_file = "Gragh_Refuting_new.csv"

# 如果文件已存在，先删除它（以确保是一个新的结果文件）
if os.path.exists(output_file):
    os.remove(output_file)

# 进行50次循环
for i in range(1):
    print(f"Running iteration {i + 1}")

    # 调用 falsify_graph 函数
    result = falsify_graph(g_true, data[:1000], plot_histogram=True)

    # 将 result 转换为 DataFrame（如果 result 是字典，直接用 DataFrame(result, index=[0]) 即可）
    result_df = pd.DataFrame([result])  # 添加 [result] 以确保是 DataFrame 格式

    # 追加保存到CSV文件
    result_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
    print(f"Iteration {i + 1} result saved to '{output_file}'")

# result = falsify_graph(g_true, data, plot_histogram=True, suggestions=True)
# print(result)
