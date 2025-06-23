# author:"flt"
# data:12/23/2024 11:39 PM
import pandas as pd
import csv
import time
from io import StringIO
import ast
import numpy as np
import networkx as nx
from dowhy import gcm
import numpy as np, pandas as pd, networkx as nx
from dowhy import gcm
np.random.seed(10)  # to reproduce these results

# output_file = 'Arrow_strength.csv'
# anomous_file = pd.read_csv('output_narrow1dot5_anomous.csv')
# df1 = pd.read_csv('output_narrow1dot5_new.csv')
# # 创建输出CSV文件的头
# with open(output_file, mode='w', newline='') as outfile:
#     fieldnames = ['index', 'param', 'value']
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     writer.writeheader()
#
# # 遍历前500条数据并进行因果归因
# for idx, row in anomous_file.iloc[210:2200].iterrows():
#     if row['crash_vehicle'] == 1:
#         traffic_density = row['traffic_density']
#         num_scenarios = row['num_scenarios']
#         accident_prob = row['accident_prob']
#         map = row['map']
#         daytime = row['daytime']
#         crash_vehicle = row['crash_vehicle']
#         crash_object = row['crash_object']
#         crash_building = row['crash_building']
#         crash_human = row['crash_human']
#         crash_sidewalk = row['crash_sidewalk']
#         out_of_road = row['out_of_road']
#         lidar_num_lasers = row['lidar_num_lasers']
#         lidar_distance = row['lidar_distance']
#         lidar_gaussian_noise = row['lidar_gaussian_noise']
#         lidar_dropout_prob = row['lidar_dropout_prob']
#         side_detector_num_lasers = row['side_detector_num_lasers']
#         side_detector_distance = row['side_detector_distance']
#         side_detector_gaussian_noise = row['side_detector_gaussian_noise']
#         side_detector_dropout_prob = row['side_detector_dropout_prob']
#         lane_line_detector_num_lasers = row['lane_line_detector_num_lasers']
#         lane_line_detector_distance = row['lane_line_detector_distance']
#         lane_line_detector_gaussian_noise = row['lane_line_detector_gaussian_noise']
#         lane_line_detector_dropout_prob = row['lane_line_detector_dropout_prob']
        # Load the data


df = pd.read_csv('output_narrow1dot5_new.csv')
        # # 第三步：因果关系建模及自动化归因
causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph(
            [('side_detector_num_lasers', 'out_of_road'), ('map', 'crash_building'),
             ('num_scenarios', 'crash_human'), ('lane_line_detector_distance', 'crash_object'),
             ('lane_line_detector_dropout_prob', 'out_of_road'), ('accident_prob', 'crash_sidewalk'),
             ('map', 'crash_object'), ('traffic_density', 'crash_human'),
             ('lane_line_detector_gaussian_noise', 'crash_human'), ('traffic_density', 'crash_building'),
             ('map', 'crash_vehicle'), ('lidar_dropout_prob', 'crash_building'), ('daytime', 'crash_human'),
             ('traffic_density', 'crash_object'), ('side_detector_gaussian_noise', 'crash_building'),
             ('lane_line_detector_gaussian_noise', 'crash_object'), ('accident_prob', 'crash_human'),
             ('lidar_dropout_prob', 'crash_object'), ('accident_prob', 'crash_building'),
             ('lane_line_detector_num_lasers', 'out_of_road'), ('traffic_density', 'crash_sidewalk'),
             ('lane_line_detector_dropout_prob', 'crash_building'),
             ('side_detector_gaussian_noise', 'crash_object'), ('lidar_dropout_prob', 'crash_sidewalk'),
             ('traffic_density', 'crash_vehicle'), ('accident_prob', 'crash_object'),
             ('lidar_gaussian_noise', 'crash_human'), ('lane_line_detector_dropout_prob', 'crash_object'),
             ('num_scenarios', 'out_of_road'), ('lidar_gaussian_noise', 'crash_building'),
             ('lidar_dropout_prob', 'crash_vehicle'), ('lidar_num_lasers', 'crash_building'),
             ('side_detector_gaussian_noise', 'crash_vehicle'),
             ('lane_line_detector_dropout_prob', 'crash_sidewalk'), ('lidar_gaussian_noise', 'crash_object'),
             ('lidar_num_lasers', 'crash_object'), ('accident_prob', 'crash_vehicle'),
             ('side_detector_dropout_prob', 'crash_human'), ('side_detector_dropout_prob', 'crash_building'),
             ('lidar_gaussian_noise', 'crash_sidewalk'), ('lidar_dropout_prob', 'crash_human'),
             ('side_detector_gaussian_noise', 'crash_human'), ('side_detector_dropout_prob', 'crash_object'),
             ('lane_line_detector_num_lasers', 'crash_building'), ('side_detector_num_lasers', 'crash_human'),
             ('side_detector_num_lasers', 'crash_building'), ('lane_line_detector_dropout_prob', 'crash_human'),
             ('lane_line_detector_distance', 'out_of_road'), ('lane_line_detector_num_lasers', 'crash_sidewalk'),
             ('side_detector_num_lasers', 'crash_object'), ('lidar_num_lasers', 'crash_human'),
             ('side_detector_num_lasers', 'crash_sidewalk'), ('traffic_density', 'out_of_road'),
             ('lane_line_detector_gaussian_noise', 'out_of_road'), ('lane_line_detector_num_lasers', 'crash_human'),
             ('daytime', 'out_of_road'), ('lane_line_detector_num_lasers', 'crash_object'),
             ('side_detector_distance', 'crash_human'), ('num_scenarios', 'crash_building'),
             ('side_detector_distance', 'crash_building'), ('lidar_gaussian_noise', 'out_of_road'),
             ('num_scenarios', 'crash_object'), ('lidar_num_lasers', 'out_of_road'),
             ('num_scenarios', 'crash_sidewalk'), ('side_detector_distance', 'crash_sidewalk'),
             ('lane_line_detector_gaussian_noise', 'crash_building'), ('num_scenarios', 'crash_vehicle'),
             ('side_detector_distance', 'crash_vehicle'), ('side_detector_dropout_prob', 'out_of_road'),
             ('daytime', 'crash_building'), ('lidar_distance', 'crash_human'), ('lidar_distance', 'crash_building'),
             ('lane_line_detector_distance', 'crash_human'), ('lane_line_detector_distance', 'crash_building'),
             ('side_detector_gaussian_noise', 'out_of_road'),
             ('lane_line_detector_gaussian_noise', 'crash_sidewalk'), ('lidar_distance', 'crash_object'),
             ('map', 'crash_human')]
        ))

gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)
# Record start time
start_time = time.time()
strength = gcm.arrow_strength(causal_model, 'crash_vehicle')
# print(strength)
# 只保留 'crash_vehicle' 的数据
filtered_data = {param: value for (param, vehicle), value in strength.items() if vehicle == 'crash_vehicle'}

# 按照值从大到小排序
sorted_data = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)

# 将结果保存为 CSV 文件
csv_filename = 'sorted_strength4.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Parameter', 'Value'])  # 写入表头

    # 写入排序后的数据
    for param, value in sorted_data:
        writer.writerow([param, value])

end_time = time.time()

# Calculate runtime
runtime = end_time - start_time

# Save runtime to Test3_1.csv
with open('Test3_1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Runtime (seconds)'])
    writer.writerow([runtime])

print(f"Sorted data saved to {csv_filename}")
print(f"Runtime saved to Test3_1.csv: {runtime} seconds")
        # # 初始化输入数据
        # anomalous_data = pd.DataFrame(
        #     data=dict(traffic_density=[traffic_density], num_scenarios=[num_scenarios], accident_prob=[accident_prob],
        #               map=[map], daytime=[daytime], crash_vehicle=[crash_vehicle], crash_object=[crash_object],
        #               crash_building=[crash_building], crash_human=[crash_human], crash_sidewalk=[crash_sidewalk],
        #               out_of_road=[out_of_road], lidar_num_lasers=[lidar_num_lasers], lidar_distance=[lidar_distance],
        #               lidar_gaussian_noise=[lidar_gaussian_noise], lidar_dropout_prob=[lidar_dropout_prob],
        #               side_detector_num_lasers=[side_detector_num_lasers],
        #               side_detector_distance=[side_detector_distance], side_detector_gaussian_noise=[side_detector_gaussian_noise],
        #               side_detector_dropout_prob=[side_detector_dropout_prob],  lane_line_detector_num_lasers=[lane_line_detector_num_lasers],
        #               lane_line_detector_distance=[lane_line_detector_distance],
        #               lane_line_detector_gaussian_noise=[lane_line_detector_gaussian_noise],
        #               lane_line_detector_dropout_prob=[lane_line_detector_dropout_prob],
        #               ))
        # with open('causal_model.pkl', 'rb') as f:
        #     loaded_model = pickle.load(f)
        # 自动化归因并返回最重要的三个参数
        # attribution_scores = gcm.attribute_anomalies(causal_model, 'crash_vehicle', anomaly_samples=anomalous_data,
        #                                              anomaly_scorer=MedianCDFQuantileScorer(),
        #                                              attribute_mean_deviation=True)
        # #排除前三的因素中有crash_vehicle的情况
        # for key in list(attribution_scores.keys()):
        #     if key == 'crash_vehicle':
        #         del attribution_scores[key]
        # # 提取并将结果写入CSV文件
        # top_7 = sorted(attribution_scores.items(), key=lambda x: x[1][0], reverse=True)[:7]
        #
        # with open(output_file, mode='a', newline='') as outfile:
        #     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        #     for param, value in top_7:
        #         writer.writerow({'index': idx, 'param': param, 'value': value[0]})