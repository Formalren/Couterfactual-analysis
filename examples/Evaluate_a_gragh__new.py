# author:"flt"
# data:10/26/2024 3:53 PM
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm

# Load the data
df = pd.read_csv('output_narrow1dot5_new.csv')

# Define the causal model
causal_model = gcm.StructuralCausalModel(nx.DiGraph(
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

# Loop 50 times
# for i in range(1):
  # Assign causal mechanisms
summary_auto_assignment = gcm.auto.assign_causal_mechanisms(causal_model, df[:1000])
print(summary_auto_assignment)

  # Fit the causal model
gcm.fit(causal_model, df)

  # Evaluate the causal model
summary_evaluation = gcm.evaluate_causal_model(causal_model, df[:1000], compare_mechanism_baselines=True)
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
