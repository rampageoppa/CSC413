import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class NuscenesDataset(Dataset):

    def __init__(self,
                root: str,
                split: str,
                transform: Optional[Callable] = None,
                local_radius: float = 50) -> None:

        self._split = split
        self._local_radius = local_radius


        if split == 'train':
            self._directory = 'train'
        elif split == 'trainval':
            self._directory = 'trainval'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0].split('-')[1] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(NuscenesDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    
    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_nuscenes(self._split, raw_path)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])

def process_nuscenes(split, raw_path):
    with open(raw_path, 'rb') as handle:
        data = pickle.load(handle)
    
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
    
    av_index = 0
    agent_index = 0
    
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    padding_mask[0, 0:50] = False
    xy = np.vstack([data['ego_hist'], data['ego_fut']])
    x[0, :] = torch.matmul(torch.tensor(xy) - origin, rotate_mat)
    heading_vector = x[0, 19] - x[0, 18]
    rotate_angles[0] = torch.atan2(heading_vector[1], heading_vector[0])
    for i, index in enumerate(agent_indices):
        i += 1
        padding_mask[i, 0:50] = False
        if flag == True:
            xy = np.vstack([data['ego_hist'], data['ego_fut']])
        else:
            xy = np.vstack([data['agent_hist'][index][:,:2], data['agent_fut'][index]])
        x[i, :] = torch.matmul(torch.tensor(xy) - origin, rotate_mat)
        heading_vector = x[i, 19] - x[i, 18]
        rotate_angles[i] = torch.atan2(heading_vector[1], heading_vector[0])
    
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(num_nodes, 2)

    ego_pos = np.vstack([data['ego_hist'], data['ego_fut']])
    ego_pos = ego_pos.reshape(1, *ego_pos.shape)
    if flag == False:
        all_agent_pos = np.concatenate((data['agent_hist'][:,:,:2], data['agent_fut']), axis=1)[agent_indices]
        all_node_pos = np.vstack([ego_pos, all_agent_pos])
    else:
        all_node_pos = np.vstack([ego_pos, ego_pos])
    node_positions_19 = torch.tensor(all_node_pos[:,19,:])

    (lane_vectors, lane_actor_index, lane_actor_vectors) = get_lane_features(np.arange(0, all_node_pos.shape[0]), node_positions_19, origin, rotate_mat, data['predicted_map'])

    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0].split('-')[1]

    # MapTR
    bev_embed = torch.tensor(data['predicted_map']['bev_features'].reshape(200, 100, 256))
    # visualize_bev(data)




    # Stream
    # Note preicison is lost when converting to tensor, might be a problem: float64 -> float32
    # version_3 (final)
    # bev_features = np.transpose(data['predicted_map']['bev_features'], (1, 2, 0))      # 100, 50, 512
    # bev_embed = torch.tensor(np.flip(bev_features, axis=0).copy())                     # 100, 50, 512

    # version 4
    # bev_features = np.transpose(data['predicted_map']['bev_features'], (1, 2, 0)) 
    # bev_flip = np.flip(bev_features, axis=0)
    # bev_embed = torch.tensor(np.flip(bev_flip, axis=1).copy())
    # breakpoint()

    # version_1
    # bev_embed = np.transpose(data['predicted_map']['bev_features'], (1, 2, 0)) 
    # version_2
    # bev_embed = torch.tensor(np.flip(np.transpose(data['predicted_map']['bev_features'], (1, 2, 0)), axis=1).copy())    # shape = 100, 50, 512


    # for index in agent_indices_original:
    #     n_point = normalize_point(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), data['agent_hist'][index, -1, :2],  data['ego_heading'].item())
    #     if -29.99 < n_point[0] < 29.99 and -14.99 < n_point[1] < 14.99:
    #         pass
    #     elif -30.1 < n_point[0] < -30:
    #         breakpoint()

    return {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'origin': origin.unsqueeze(0),
        'theta': theta,
        'bev_embed': bev_embed,
    }


def get_lane_features(node_inds,
                      node_positions, 
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      predicted_map):
    lane_positions, lane_vectors = [], []
    lane_position_betas, lane_vector_betas = [], []
    lane_ids = set()

    boundary_scores = predicted_map['boundary_scores']
    boundary_indices = np.where(boundary_scores > 0)[0]  #0.4
    boundary = np.array(predicted_map['boundary'])[boundary_indices]

    divider_scores = predicted_map['divider_scores']
    divider_indices = np.where(divider_scores > 0)[0]
    divider = np.array(predicted_map['divider'])[divider_indices]

    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    node_positions = torch.cat([node_positions, torch.zeros_like(node_positions)], dim=-1)

    if boundary.shape[0] == 0:
        centerlines = divider
        centerline_betas = predicted_map['divider_betas'][divider_indices]
    else:
        centerlines = boundary
        centerline_betas = predicted_map['boundary_betas'][boundary_indices]

    for i, centerline in enumerate(centerlines):

        lane_centerline = torch.matmul(torch.tensor(centerline, dtype=torch.float32) - origin, rotate_mat)
        
        # lane_positions.append(lane_centerline[:-1])
        # lane_position_betas.append(torch.tensor(centerline_betas[i][:-1]))
        lane_positions.append(lane_centerline)
        lane_position_betas.append(torch.tensor(centerline_betas[i]))

        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        lane_vector_betas.append(torch.tensor(centerline_betas[i][1:]) + torch.tensor(centerline_betas[i][:-1]))
        count = len(lane_centerline) - 1

    try:
        lane_positions = torch.cat(lane_positions, dim=0)
        lane_position_betas = torch.cat(lane_position_betas, dim=0)
        lane_positions = torch.cat((lane_positions, lane_position_betas), dim=-1)

    except:
        breakpoint()
    # lane_vectors = torch.cat(lane_vectors, dim=0)
    # lane_vector_betas = torch.cat(lane_vector_betas, dim=0)
    # lane_vectors = torch.cat((lane_vectors, lane_vector_betas), dim=-1)

    # lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    # lane_actor_vectors = \
    #     lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_positions.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_positions.size(0), 1)

    return lane_positions, lane_actor_index, lane_actor_vectors


def normalize_lanes(x, y, lanes, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    n_lanes = []
    for lane in lanes:
        normalize_lane = np.dot(lane[:, :2] - np.array([x, y]), R)
        n_lanes.append(normalize_lane)
    
    return n_lanes

def normalize_point(x, y, point, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    normalize_point = np.dot(point - np.array([x, y]), R)
    
    return normalize_point

def normalize_traj(x, y, traj, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    normalize_traj = np.dot(traj[:, :2] - np.array([x, y]), R)
    
    return normalize_traj


def visualize_bev(data):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    feature_embedding = data['predicted_map']['bev_features'].reshape(200, 100, 256)
    # aggregated_embedding = np.mean(feature_embedding, axis=2)

    reshaped_embedding = feature_embedding.reshape(-1, 256)
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(reshaped_embedding)
    pca_image = principal_component.reshape(200, 100)
    aggregated_embedding = pca_image

    normalized_embedding = (aggregated_embedding - np.min(aggregated_embedding)) / (np.max(aggregated_embedding) - np.min(aggregated_embedding))
    normalized_embedding *= 255

    # plt.imshow(normalized_embedding, cmap='gray')
    # plt.colorbar()
    # plt.title("Visualization of Feature Embedding")
    # plt.show()
    axes[0].imshow(normalized_embedding, cmap='gray')
    axes[0].set_title("Visualization of Feature Embedding")
    axes[0].axis('off')  # Turn off axis

    value = data
    centerlines = value['gt_map']['centerlines']
    left_edges = value['gt_map']['left_edges']
    right_edges = value['gt_map']['right_edges']

    boundary_scores = value['predicted_map']['boundary_scores']
    ped_crossing_scores = value['predicted_map']['ped_crossing_scores']
    boundary_indices = np.where(boundary_scores > 0.4)[0]
    ped_crossing_indices = np.where(ped_crossing_scores > 0.4)[0]
    boundary = np.array(value['predicted_map']['boundary'])[boundary_indices]
    ped_crossing = np.array(value['predicted_map']['ped_crossing'])[ped_crossing_indices]

    n_centerlines = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), centerlines, value['ego_heading'].item())
    n_left_edges = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), left_edges, value['ego_heading'].item())
    n_right_edges = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), right_edges, value['ego_heading'].item())

    n_boundary = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), boundary, value['ego_heading'].item())
    n_ped_crossing = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), ped_crossing, value['ego_heading'].item())

    # fig, ax = plt.subplots(figsize=(6, 12))

    for lane in n_centerlines:
        axes[1].plot(lane[:, 1], lane[:, 0], c='blue')
    
    for lane in n_boundary:
        axes[1].plot(lane[:, 1], lane[:, 0], c='green')

    for lane in n_ped_crossing:
        axes[1].plot(lane[:, 1], lane[:, 0], c='black')
    
    axes[1].scatter(0, 0, s=100, c='red', marker="*")

    n_ego_hist = normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['ego_hist'], value['ego_heading'].item())
    n_ego_fut =  normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['ego_fut'], value['ego_heading'].item())
    axes[1].plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='orange')
    axes[1].plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

    for i in range(value['agent_hist'].shape[0]):
        color = {1: 'black', 2: 'orange', 3: 'black', 4: 'black'}
        n_point = normalize_point(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['agent_hist'][i, -1, :2],  value['ego_heading'].item())
        if -30 <= n_point[0] <= 30 and -15 <= n_point[1] <= 15:
            axes[1].scatter(n_point[1], n_point[0], s=100, c=color[value['agent_type'][i].item()], marker="*")
        else:
            continue
    
    axes[1].invert_xaxis()

    # Adjust the layout
    plt.tight_layout() 
    plt.show()