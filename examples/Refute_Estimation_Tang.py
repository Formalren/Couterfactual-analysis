import json
import random
import csv
import time

import pydot
import dowhy.gcm as gcm
import statistics
import numpy as np
from causallearn.search.FCMBased import lingam
from networkx.drawing.nx_pydot import from_pydot
from causallearn.search.FCMBased.lingam.utils import make_dot
from dowhy import CausalModel
import pandas as pd




import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



def read_data(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)

    data_list = []
    scenario_id = []
    for item in data.items():
        scenario_id.append(item[0])
        temp_scenario = []
        for i in range(len(item[1]['data'])):
            sce = item[1]['data'][str(i)]
            temp_scene = []
            temp_scene.extend(sce['npc_representation'])
            temp_scene.extend(sce['act_representation'])
            temp_scene.extend(sce['failure_representation'])
            temp_scenario.append(temp_scene)
        data_list.append(temp_scenario)
    return data_list, scenario_id



def get_edges_remove(scene_variables, action_variables, fault_variables):
    """This function is used to exclude edges which are not possible"""
    all_variable_names = scene_variables + action_variables + fault_variables
    tabu_edges = [[-1 for _ in range(len(all_variable_names))] for _ in range(len(all_variable_names))]
    # for id_variable in range(len(all_variable_names)):
    #     tabu_edges[id_variable][id_variable] = 0
    for id_variable in range(len(scene_variables)):
        for id_variable1 in range(len(scene_variables)):
            tabu_edges[id_variable1][id_variable] = 0

    for id_variable in range(len(scene_variables),len(scene_variables+action_variables)):
        for id_variable1 in range(len(scene_variables+action_variables)):
            tabu_edges[id_variable1][id_variable] = 0

    for id_variable in range(len(scene_variables+action_variables),len(all_variable_names)):
        for id_variable1 in range(len(all_variable_names)):
            tabu_edges[id_variable1][id_variable] = 0

    return tabu_edges

def get_causal_graph(data_array, bk):

    #  discovery causal graph using FCI
    last_two_columns =data_array[:, -2:]
    has_one = np.any(last_two_columns == 1)
    model_lingam = lingam.DirectLiNGAM()
    model_lingam.fit(data_array)

    # the adjacency metrix of causal graph
    adjacency_matrix = model_lingam.adjacency_matrix_
    causal_order = model_lingam.causal_order_



    # add the causes of violation
    if has_one:
        fault_index = np.where(last_two_columns == 1)[0][0]
        if np.where(last_two_columns == 1)[1][0] == 0:
            adjacency_matrix[-2][:-2] = data_array[fault_index][:-2]
        else:
            adjacency_matrix[-1][:-2] = data_array[fault_index][:-2]

    # calculate the causal strength
    causal_effect_strength = []
    for i in causal_order:
        temp_effect = []
        for j in causal_order:
            temp_effect.append(model_lingam.estimate_total_effect(data_array, i, j))
        causal_effect_strength.append(temp_effect)

    filter_matrix = [[1 if x == -1 else x for x in row] for row in bk]
    causal_strength_matrix = [[a * b for a, b in zip(row1, row2)] for row1, row2 in zip(filter_matrix, causal_effect_strength)]


    return adjacency_matrix, causal_strength_matrix, has_one



def from_matrix_to_edge(matrix, variable_names):
    temp_edges = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] != 0:
                temp_edges.append((variable_names[j], variable_names[i]))
    return temp_edges



def render_dot_to_png(dot_source, output_file):
    # Create Digraph object from Dot source
    # dot = graphviz.Source(dot_source)

    # Get list of nodes and edges
    nodes = dot_source.body[:]
    edges = [e for e in dot_source.body if '->' in e]

    # Determine nodes with edges
    nodes_with_edges = set()
    for edge in edges:
        src, dest = edge.split('->')
        nodes_with_edges.add(src.strip())
        nodes_with_edges.add(dest.strip())

    # Remove nodes without edges
    nodes_to_remove = [node for node in nodes if node.split()[0] not in nodes_with_edges]
    for node in nodes_to_remove:
        dot_source.body.remove(node)

    # Save Dot graph as PNG
    dot_source.format = 'png'
    dot_source.render(output_file, view=False)



# 写入日志的函数
def log_to_txt(file_name, message):
    with open(file_name, 'a') as f:  # 以追加模式打开文件
        f.write(message + '\n')      # 写入消息并换行



if __name__ == '__main__':
    data_path = 'causal_data.json'
    data_list, scenario_id = read_data(data_path)
    s_list = [f'S{i}' for i in range(32)]
    a_list = [f'A{i}' for i in range(5)]
    f_list = [f'F{i}' for i in range(2)]
    variable_names = s_list + a_list + f_list

    pk = get_edges_remove(s_list, a_list, f_list)

    failure_id_list = []
    consistent_rate = []
    PMC_p_value = []
    LMC_p_value = []

    for i in range(10):
        data_array = np.array(data_list[i])
        new_data_array = data_array[:-2, :-2]
        data_pd = pd.DataFrame(new_data_array, columns=variable_names[:-2])
        # print(data_pd)

        model = CausalModel(
            data=data_pd,
            treatment=s_list,
            outcome=a_list,
            common_causes=[]
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # print(identified_estimand)
        causal_estimate_reg = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.linear_regression",
                                                    test_significance=True)
        # print(causal_estimate_reg)

        res_random = model.refute_estimate(identified_estimand, causal_estimate_reg, method_name="random_common_cause")
        res_removing = model.refute_estimate(identified_estimand, causal_estimate_reg, method_name="data_subset_refuter", show_progress_bar=True, subset_fraction=0.9)
        # res_random = model.refute_estimate(identified_estimand, causal_estimate_reg, method_name="add_unobserved_common_cause",
        #                                    confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linea")
        print(res_random, res_removing)
        # time.sleep(1)
        # print(res_random)
        # print(type(res_random))
        # log_to_txt('output.txt', res_random)

