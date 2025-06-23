# author:"flt"
# data:10/28/2024 10:38 PM
import numpy as np

from dowhy import CausalModel
import dowhy.datasets
import pandas as pd
df = pd.read_csv('output_narrow1dot5.csv')
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
    graph="GML_13270.gml",
    common_causes=[]
)

from IPython.display import Image, display
display(Image(filename="causal_model.png"))

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
print(estimate)
print("Causal Estimate is " + str(estimate.value))

res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause", show_progress_bar=True)
print(res_random)

res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", show_progress_bar=True, subset_fraction=0.9)
print(res_subset)