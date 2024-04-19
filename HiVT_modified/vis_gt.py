import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse

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


folder_path = "/home/guxunjia/project/HiVT_data_gt_old/val/data/"
data = {}

for filename in os.listdir(folder_path):
    # breakpoint()
    if filename.endswith('.pkl'):
        scene_id = filename.replace('scene-', '').replace('.pkl', '')

        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'rb') as file:
            data_content = pickle.load(file)
        
        data[int(scene_id)] = data_content

with open('/home/guxunjia/project/HiVT_modified/results/old/maptrv2/result_gt.pkl', 'rb') as handle:
    predicted_data = pickle.load(handle)

for key, value in predicted_data.items():
    try:
        data[key]['predict_fut'] = value
    except:
        breakpoint()


for key, value in data.items():
    # centerlines = value['gt_map']['centerlines']
    # try:
    #     left_edges = value['gt_map']['left_edges']
    # except:
    #     breakpoint()
    # right_edges = value['gt_map']['right_edges']
    centerlines = value['vec_map']['centerlines']
    try:
        left_edges = value['vec_map']['left_edges']
    except:
        breakpoint()
    right_edges = value['vec_map']['right_edges']

    x, y = value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item()
    heading = value['ego_heading'].item()
    n_centerlines = normalize_lanes(x, y, centerlines, heading)
    n_left_edges = normalize_lanes(x, y, left_edges, heading)
    n_right_edges = normalize_lanes(x, y, right_edges, heading)

    fig, ax = plt.subplots(figsize=(6, 12))

    for lane in n_centerlines:
        plt.plot(lane[:, 1], lane[:, 0], c='grey', linestyle='--')
    
    for lane in n_left_edges:
        plt.plot(lane[:, 1], lane[:, 0], c='black')

    for lane in n_right_edges:
        plt.plot(lane[:, 1], lane[:, 0], c='black')
    
    plt.scatter(0, 0, s=100, c='red', marker="*")

    n_ego_hist = normalize_traj(x, y, value['ego_hist'], heading)
    n_ego_fut =  normalize_traj(x, y, value['ego_fut'], heading)
    plt.plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='blue')
    plt.plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

    for i in range(6):
        n_predict_fut = normalize_traj(x, y, value['predict_fut'][i], heading)
        # n_predict_fut = value['predict_fut'][i]
        plt.plot(n_predict_fut[:, 1], n_predict_fut[:, 0], c='orange')
        # breakpoint()

        beta_x = value['predict_fut'][i][:, 2]
        beta_y = value['predict_fut'][i][:, 3]

        # Calculate the variance from the beta values
        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
            
        # Draw an axis-aligned ellipse around each point
        for j in range(len(beta_x)):
            # Using variance to set the width and height of the ellipse
            ellipse = Ellipse((n_predict_fut[j][1], n_predict_fut[j][0]), width=np.sqrt(var_x[j])*2, height=np.sqrt(var_y[j])*2,
                            edgecolor='red', fc='None', lw=2)
            ax.add_patch(ellipse)

    for i in range(value['agent_hist'].shape[0]):
        color = {1: 'black', 2: 'orange', 3: 'black', 4: 'black'}
        n_point = normalize_point(x, y, value['agent_hist'][i, -1, :2],  heading)
        if -30 <= n_point[0] <= 30 and -15 <= n_point[1] <= 15:
            plt.scatter(n_point[1], n_point[0], s=100, c=color[value['agent_type'][i].item()], marker="*")
        else:
            continue

    # print(value['sample_token'])
    # print(value['scene_name'])

    ax.invert_xaxis()
    plt.show()
    # plt.savefig('/home/guxunjia/project/HiVT_modified/gt_plots/' + value['scene_name'] + '.png', dpi=300)
    plt.close(fig)

