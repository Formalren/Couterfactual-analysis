# author:"flt"
# data:10/14/2024 8:49 PM
import os, sys

sys.path.append(os.path.abspath("../../../"))
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
import dowhy.datasets

# Config dict to set the logging level
import logging.config

# DEFAULT_LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'loggers': {
#         '': {
#             'level': 'ERROR',
#         },
#     }
# }

# logging.config.dictConfig(DEFAULT_LOGGING)
# Disabling warnings output
import warnings
from sklearn.exceptions import DataConversionWarning

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
df = pd.read_csv('output_end1.csv')
# axis表示删除df中的某一列，axis为0表示删除df中的某一行
# df = df.drop("traffic_density", axis=1)
graph_str = [('lane_line_detector_num_lasers', 'crash_human'), ('num_scenarios', 'crash_object'),
             ('lidar_dropout_prob', 'crash_building'), ('lane_line_detector_num_lasers', 'crash_object'),
             ('side_detector_distance', 'crash_sidewalk'), ('num_scenarios', 'out_of_road'),
             ('lidar_distance', 'out_of_road'),
             ('lane_line_detector_num_lasers', 'out_of_road'), ('lidar_gaussian_noise', 'crash_vehicle'),
             ('lane_line_detector_distance', 'crash_human'),
             ('lane_line_detector_gaussian_noise', 'crash_building'),
             ('num_scenarios', 'crash_sidewalk'), ('lane_line_detector_distance', 'crash_object'),
             ('lidar_distance', 'crash_sidewalk'), ('lane_line_detector_num_lasers', 'crash_sidewalk'),
             ('lane_line_detector_distance', 'out_of_road'), ('lane_line_detector_dropout_prob', 'crash_building'),
             ('lidar_distance', 'crash_vehicle'), ('side_detector_gaussian_noise', 'crash_human'),
             ('lane_line_detector_distance', 'crash_sidewalk'),
             # ('traffic_density', 'crash_human'),
             ('num_scenarios', 'crash_human'), ('side_detector_num_lasers', 'crash_human'), ('map', 'crash_object'),
             ('side_detector_num_lasers', 'crash_object'), ('accident_prob', 'crash_human'),
             # ('traffic_density', 'out_of_road'),
             ('lidar_gaussian_noise', 'crash_building'), ('side_detector_gaussian_noise', 'crash_sidewalk'),
             ('accident_prob', 'crash_object'), ('side_detector_distance', 'crash_vehicle'),
             ('lidar_dropout_prob', 'crash_human'),
             # ('traffic_density', 'crash_sidewalk'),
             ('lidar_num_lasers', 'crash_building'),
             ('side_detector_num_lasers', 'crash_sidewalk'), ('daytime', 'crash_human'),
             ('num_scenarios', 'crash_vehicle'),
             ('lidar_distance', 'crash_building'), ('lidar_dropout_prob', 'out_of_road'),
             ('accident_prob', 'crash_sidewalk'),
             ('lane_line_detector_num_lasers', 'crash_vehicle'), ('daytime', 'out_of_road'),
             ('lane_line_detector_gaussian_noise', 'crash_object'), ('lidar_dropout_prob', 'crash_sidewalk'),
             ('lane_line_detector_gaussian_noise', 'out_of_road'), ('lane_line_detector_distance', 'crash_vehicle'),
             ('map', 'crash_human'), ('daytime', 'crash_sidewalk'), ('side_detector_dropout_prob', 'crash_human'),
             ('lane_line_detector_gaussian_noise', 'crash_sidewalk'),
             ('side_detector_dropout_prob', 'crash_object'),
             ('map', 'out_of_road'), ('side_detector_distance', 'crash_building'),
             ('side_detector_dropout_prob', 'out_of_road'),
             ('map', 'crash_sidewalk'), ('lidar_gaussian_noise', 'crash_human'),
             ('side_detector_dropout_prob', 'crash_sidewalk'),
             # ('traffic_density', 'crash_vehicle'),
             ('num_scenarios', 'crash_building'), ('map', 'crash_vehicle'),
             ('side_detector_num_lasers', 'crash_vehicle'), ('lane_line_detector_num_lasers', 'crash_building'),
             ('lidar_gaussian_noise', 'crash_object'), ('lane_line_detector_gaussian_noise', 'crash_human'),
             ('lidar_num_lasers', 'crash_human'), ('lidar_gaussian_noise', 'out_of_road'),
             ('accident_prob', 'crash_vehicle'),
             ('lane_line_detector_dropout_prob', 'crash_human'), ('lidar_num_lasers', 'crash_object'),
             ('lane_line_detector_distance', 'crash_building'), ('lidar_num_lasers', 'out_of_road'),
             ('lidar_gaussian_noise', 'crash_sidewalk'), ('lidar_distance', 'crash_object'),
             ('lidar_dropout_prob', 'crash_vehicle'),
             ('daytime', 'crash_building'), ('daytime', 'crash_vehicle'), ('lidar_num_lasers', 'crash_sidewalk'),
             ('lane_line_detector_dropout_prob', 'crash_sidewalk'),
             ('side_detector_gaussian_noise', 'crash_building'),
             # ('traffic_density', 'crash_building'),
             ('side_detector_distance', 'crash_human'),
             ('map', 'crash_building'),
             ('side_detector_num_lasers', 'crash_building'), ('side_detector_dropout_prob', 'crash_building'),
             ('side_detector_dropout_prob', 'crash_vehicle'), ('accident_prob', 'crash_building'),
             ('side_detector_distance', 'out_of_road'), ('lidar_distance', 'crash_human')]
model = CausalModel(
    data=df,
    treatment=['traffic_density', 'num_scenarios', 'accident_prob', 'map', 'daytime', 'lidar_num_lasers',
               'lidar_distance',
               'lidar_gaussian_noise', 'lidar_dropout_prob', 'side_detector_num_lasers', 'side_detector_distance',
               'side_detector_gaussian_noise',
               'side_detector_dropout_prob',
               'lane_line_detector_num_lasers',
               'lane_line_detector_distance',
               'lane_line_detector_gaussian_noise',
               'lane_line_detector_dropout_prob'],
    outcome=['crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'crash_sidewalk', 'out_of_road'],
    graph="falsify_g_true6.gml",
    # test_significance=None,
    common_causes=['traffic_density']
)
model.view_model()
from IPython.display import Image, display

display(Image(filename="causal_model.png"))
df.head()
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print(estimate)
data = df['traffic_density']
refute = model.refute_estimate(identified_estimand, estimate,
                               method_name = "add_unobserved_common_cause",
                               simulation_method="linear-partial-R2",
                               benchmark_common_causes=[],
                               effect_fraction_on_treatment=[1]
                               )
print(refute.stats)
