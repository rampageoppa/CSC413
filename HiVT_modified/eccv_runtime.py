import pickle 
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# with open('/home/guxunjia/project/HiVT_modified/runtime_results/map/maptrv2_cent.pkl', 'rb') as handle:
#     data = pickle.load(handle)

# hivt_cpu_avg = 0 
# hivt_cuda_avg = 0 
# local_cpu_avg = 0
# local_cuda_avg = 0
# for i in range(len(data)):
#     hivt_cpu_avg += data[i]['hivt_forward']['cpu_time']
#     hivt_cuda_avg += data[i]['hivt_forward']['cuda_time']
#     local_cpu_avg += data[i]['local_forward']['cpu_time']
#     local_cuda_avg += data[i]['local_forward']['cuda_time']

# print(f'HiVT Forward CPU MEAN: {hivt_cpu_avg / len(data)}')
# print(f'HiVT Forward CUDA MEAN: {hivt_cuda_avg / len(data)}')
# print(f'Local Forward CPU MEAN: {local_cpu_avg / len(data)}')
# print(f'Local Forward CUDA MEAN: {local_cuda_avg / len(data)}')


# entire_cpu_avg = 0 
# entire_cuda_avg = 0 
# bev_cpu_avg = 0
# bev_cuda_avg = 0
# for token, data_ in data.items():
#     # breakpoint()
#     entire_cpu_avg += data_['entire_forward']['cpu_time']
#     entire_cuda_avg += data_['entire_forward']['cuda_time']
#     bev_cpu_avg += data_['pre_bev_extraction']['cpu_time']
#     bev_cuda_avg += data_['pre_bev_extraction']['cuda_time']

# print(f'Entire Forward CPU MEAN: {entire_cpu_avg / len(data)}')
# print(f'Entire Forward CUDA MEAN: {entire_cuda_avg / len(data)}')
# print(f'BEV Extraction CPU MEAN: {bev_cpu_avg / len(data)}')
# print(f'BEV Extraction CUDA MEAN: {bev_cuda_avg / len(data)}')

map_model = 'maptrv2'
with open(f'/home/guxunjia/project/HiVT_modified/runtime_results/full/map/{map_model}.pkl', 'rb') as handle:
    map_runtime = pickle.load(handle)

with open(f'/home/guxunjia/project/HiVT_modified/runtime_results/full/hivt/{map_model}/ab.pkl', 'rb') as handle:
    hivt_runtime = pickle.load(handle)

# Assuming 'path_to_folder' is the path to the folder containing your pickle files
path_to_folder = f'/home/guxunjia/project/HiVT_data/maptrv2_2/full_val/data'
files = [f for f in os.listdir(path_to_folder) if f.startswith('scene-') and f.endswith('.pkl')]

# Initialize an empty dictionary
agent_map = {}
pair = {}

for file in files:
    # Construct the full path to the file
    file_path = os.path.join(path_to_folder, file)
    
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Assuming 'sample_token' is a key in your loaded data
    sample_token = data['sample_token']
    scene_id = int(file.split('-')[1].split('.')[0])

    origin = torch.tensor(data['ego_hist'][-1], dtype=torch.float)
    av_heading_vector = origin - torch.tensor(data['ego_hist'][-2], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    agent_indices_original = np.where(data['agent_type'] == 1)[0]
    keep_indices = []
    flag = False
    for index in agent_indices_original:
        # n_point = normalize_point(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), data['agent_hist'][index, -1, :2],  data['ego_heading'].item())
        # n_point = torch.matmul((torch.tensor(data['agent_hist'][index, -1, :2]) - origin).unsqueeze(0), rotate_mat)
        n_point = torch.matmul(torch.tensor([data['agent_hist'][index, -1, :2]]) - origin, rotate_mat)
        # breakpoint()
        if -30 < n_point[0][0].item() < 30 and -15 < n_point[0][1].item() < 15:
            keep_indices.append(index)

    if len(keep_indices) == 0:
        keep_indices.append(500)
        flag = True
    agent_indices = np.array(keep_indices)
    num_nodes = len(agent_indices) + 1

    # num_map = len(data['vec_map']['centerlines'])
    # num_map = len(data['predicted_map']['centerline'])
    num_map = len(data['predicted_map']['boundary'])

    hivt_runtime_i = hivt_runtime[scene_id]
    map_runtime_i = map_runtime[sample_token]

    if (num_nodes, num_map) in pair:
        pair[(num_nodes, num_map)]['predict_time']['hivt_forward']['cuda_time']
        pair[(num_nodes, num_map)]['predict_time']['hivt_forward']['cpu_time']
        pair[(num_nodes, num_map)]['map_time']['entire_forward']['cuda_time']
        pair[(num_nodes, num_map)]['map_time']['entire_forward']['cpu_time']
        pair[(num_nodes, num_map)]['map_time']['pre_bev_extraction']['cuda_time']
        pair[(num_nodes, num_map)]['map_time']['pre_bev_extraction']['cpu_time']
        map_time = 
    pair[(num_nodes, num_map)] = {"predict_time": hivt_runtime_i, 'map_time': map_runtime_i}

with open('/home/guxunjia/project/HiVT_modified/result.pkl', 'wb') as file:
    pickle.dump(pair, file)

breakpoint()

max_agent = 0
max_map = 0
for key, value in pair.items():
    if key[0] > max_agent:
        max_agent = key[0]
    if key[1] > max_map:
        max_map = key[1]

runtime_matrix = np.zeros((max_agent, max_map))

for key, value in pair.items():
    time_1 = value['predict_time']['hivt_forward']['cuda_time'] + value['map_time']['entire_forward']['cuda_time']
    time_2 = value['predict_time']['hivt_forward']['cuda_time'] + value['map_time']['pre_bev_extraction']['cuda_time']

    time_3 = value['predict_time']['hivt_forward']['cpu_time'] + value['map_time']['entire_forward']['cpu_time']
    time_4 = value['predict_time']['hivt_forward']['cpu_time'] + value['map_time']['pre_bev_extraction']['cpu_time']

    runtime_matrix[key[0]-1, key[1]-1] = time_2
    # runtime_matrix[key[0]-1, key[1]-1] = time_1 + time_3
    # runtime_matrix[key[0]-1, key[1]-1] = time_2 + time_4

# Stream
# runtime_matrix = runtime_matrix[:, :45]
# plt.figure(figsize=(10, 8))
plt.figure()
heatmap = plt.imshow(runtime_matrix, cmap='coolwarm', interpolation='nearest')

plt.colorbar(heatmap)
plt.gca().invert_yaxis()

# Set the tick labels
# plt.xticks(ticks=np.arange(max_map), labels=num_map_elements)
# plt.yticks(ticks=np.arange(max_agent), labels=max_agent)

# Set labels and title
plt.xlabel('Number of Map Elements', fontsize=26)
plt.ylabel('Number of Agents', fontsize=26)
plt.title('Runtime Heatmap', fontsize=26)

plt.show()
# plt.savefig('/home/guxunjia/project/HiVT_modified/run_time.pdf', dpi=600)
# plt.savefig('/home/guxunjia/project/HiVT_modified/run_time.png', dpi=600)

# breakpoint()