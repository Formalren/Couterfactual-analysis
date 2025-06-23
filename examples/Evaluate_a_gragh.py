# author:"flt"
# data:10/26/2024 3:53 PM
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm

# Load the data
df = pd.read_csv('output_narrow1dot5_new.csv')

# Define the causal model
causal_model = gcm.StructuralCausalModel(nx.DiGraph(
    [('lane_line_detector_num_lasers', 'crash_sidewalk'), ('side_detector_dropout_prob', 'crash_vehicle'),
     ('side_detector_num_lasers', 'crash_vehicle'), ('lane_line_detector_gaussian_noise', 'out_of_road'),
     ('side_detector_distance', 'crash_building'), ('lidar_gaussian_noise', 'lidar_distance'),
     ('accident_prob', 'crash_sidewalk'), ('lidar_dropout_prob', 'out_of_road'),
     ('lidar_gaussian_noise', 'crash_building'), ('num_scenarios', 'out_of_road'), ('daytime', 'lidar_distance'),
     ('side_detector_num_lasers', 'crash_sidewalk'), ('lane_line_detector_num_lasers', 'out_of_road'),
     ('map', 'lidar_distance'), ('daytime', 'crash_building'), ('lidar_num_lasers', 'out_of_road'),
     ('map', 'crash_building'), ('lidar_gaussian_noise', 'crash_human'), ('map', 'lane_line_detector_distance'),
     ('traffic_density', 'crash_vehicle'), ('side_detector_dropout_prob', 'out_of_road'),
     ('lane_line_detector_gaussian_noise', 'crash_human'), ('side_detector_distance', 'crash_vehicle'),
     ('lidar_dropout_prob', 'crash_sidewalk'), ('num_scenarios', 'crash_sidewalk'),
     ('side_detector_gaussian_noise', 'crash_human'), ('lidar_num_lasers', 'crash_sidewalk'),
     ('accident_prob', 'crash_human'), ('side_detector_dropout_prob', 'crash_sidewalk'),
     ('traffic_density', 'out_of_road'), ('side_detector_num_lasers', 'crash_human'),
     ('lane_line_detector_gaussian_noise', 'crash_building'), ('side_detector_gaussian_noise', 'lidar_distance'),
     ('lane_line_detector_gaussian_noise', 'lane_line_detector_distance'), ('side_detector_distance', 'out_of_road'),
     ('lane_line_detector_num_lasers', 'lidar_distance'), ('side_detector_gaussian_noise', 'crash_building'),
     ('lidar_gaussian_noise', 'crash_object'), ('lane_line_detector_num_lasers', 'crash_building'),
     ('lane_line_detector_num_lasers', 'lane_line_detector_distance'),
     ('lane_line_detector_dropout_prob', 'crash_sidewalk'), ('daytime', 'out_of_road'),
     ('accident_prob', 'crash_building'), ('lidar_dropout_prob', 'crash_human'), ('traffic_density', 'crash_sidewalk'),
     ('side_detector_num_lasers', 'lidar_distance'), ('num_scenarios', 'crash_human'), ('map', 'out_of_road'),
     ('map', 'crash_object'), ('lidar_gaussian_noise', 'crash_vehicle'),
     ('side_detector_dropout_prob', 'crash_building'), ('lane_line_detector_num_lasers', 'crash_human'),
     ('side_detector_dropout_prob', 'lane_line_detector_distance'), ('side_detector_num_lasers', 'crash_building'),
     ('side_detector_num_lasers', 'lane_line_detector_distance'), ('side_detector_distance', 'crash_sidewalk'),
     ('lidar_num_lasers', 'crash_human'), ('lidar_gaussian_noise', 'crash_sidewalk'), ('map', 'crash_vehicle'),
     ('side_detector_dropout_prob', 'crash_human'), ('daytime', 'crash_sidewalk'), ('map', 'crash_sidewalk'),
     ('lidar_dropout_prob', 'crash_building'), ('num_scenarios', 'crash_building'),
     ('traffic_density', 'crash_building'), ('lidar_gaussian_noise', 'out_of_road'),
     ('traffic_density', 'lane_line_detector_distance'), ('lidar_num_lasers', 'crash_building'),
     ('lidar_num_lasers', 'lane_line_detector_distance'), ('lane_line_detector_gaussian_noise', 'crash_object'),
     ('lane_line_detector_dropout_prob', 'crash_human'), ('traffic_density', 'crash_human'),
     ('num_scenarios', 'crash_object'), ('side_detector_distance', 'crash_human'),
     ('lane_line_detector_num_lasers', 'crash_object'), ('lidar_num_lasers', 'crash_object'),
     ('accident_prob', 'crash_object'), ('lidar_dropout_prob', 'crash_vehicle'), ('num_scenarios', 'crash_vehicle'),
     ('lane_line_detector_dropout_prob', 'lidar_distance'), ('lane_line_detector_gaussian_noise', 'crash_sidewalk'),
     ('side_detector_dropout_prob', 'crash_object'), ('side_detector_num_lasers', 'crash_object'),
     ('daytime', 'crash_human'), ('lane_line_detector_dropout_prob', 'crash_building'),
     ('side_detector_gaussian_noise', 'crash_sidewalk'), ('accident_prob', 'crash_vehicle'), ('map', 'crash_human')]

))

# Loop 50 times
for i in range(1):
  # Assign causal mechanisms
  summary_auto_assignment = gcm.auto.assign_causal_mechanisms(causal_model, df)
  print(summary_auto_assignment)

  # Fit the causal model
  gcm.fit(causal_model, df)

  # Evaluate the causal model
  summary_evaluation = gcm.evaluate_causal_model(causal_model, df, compare_mechanism_baselines=True)
  print(summary_evaluation)

  # Create a DataFrame for the current summaries
  summary_df = pd.DataFrame({
    'summary_auto_assignment': [summary_auto_assignment],
    'summary_evaluation': [summary_evaluation]
  })

  # Append the current summaries to CSV files
  summary_df.to_csv('gragh_assignment.csv', mode='a',
                    header=not pd.io.common.file_exists('gragh_assignment.csv'), index=False)
  summary_df.to_csv('gragh_evaluation.csv', mode='a', header=not pd.io.common.file_exists('gragh_evaluation.csv'),
                    index=False)
