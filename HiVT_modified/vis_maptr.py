import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse
from tqdm import tqdm

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


def plot_points_with_laplace_variances(x, y, beta_x, beta_y, color, sample_idx, ax, heading):
    for i in range(len(x)):
        plt.plot(x[i], y[i], color=color, linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=color, s=2, alpha=0.8, zorder=-1)

        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        
        for j in range(len(x[i])):
            ellipse = Ellipse((x[i][j], y[i][j]), width=np.sqrt(var_x[i][j])*2, height=np.sqrt(var_y[i][j])*2,
                              fc=color, lw=0.5, alpha=0.5) #alpha=2, edgecolor='red'
            ax.add_patch(ellipse)
        # ax.annotate(f"{scores[i]:.2f}", (x[i][0], y[i][0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)

folder_path = "/home/guxunjia/project/HiVT_data/maptr/mini_val/data/"
data = {}

for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        scene_id = filename.replace('scene-', '').replace('.pkl', '')

        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'rb') as file:
            data_content = pickle.load(file)
        # del data_content['predicted_map']['bev_features']
        # breakpoint()
        data[int(scene_id)] = data_content

# with open('/home/guxunjia/project/HiVT_modified/result.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/HiVT_modified/results/result_maptr_std_mini.pkl', 'rb') as handle:
    predicted_data = pickle.load(handle)

for key, value in predicted_data.items():
    try:
        data[key]['predict_fut'] = value
    except:
        breakpoint()


for key, value in tqdm(data.items()):
    x, y = value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item()
    heading = value['ego_heading'].item()
    fig, ax = plt.subplots(figsize=(2, 4))
    plt.axis('off')

    divider_scores = value['predicted_map']['divider_scores']
    ped_crossing_scores = value['predicted_map']['ped_crossing_scores']
    boundary_scores = value['predicted_map']['boundary_scores']

    divider_indices = np.where(divider_scores > 0.4)[0]
    ped_crossing_indices = np.where(ped_crossing_scores > 0.4)[0]
    boundary_indices = np.where(boundary_scores > 0.4)[0]

    divider = np.array(value['predicted_map']['divider'])[divider_indices]
    ped_crossing = np.array(value['predicted_map']['ped_crossing'])[ped_crossing_indices]
    boundary = np.array(value['predicted_map']['boundary'])[boundary_indices]

    divider_betas = value['predicted_map']['divider_betas'][divider_indices]
    ped_crossing_betas = value['predicted_map']['ped_crossing_betas'][ped_crossing_indices]
    boundary_betas = value['predicted_map']['boundary_betas'][boundary_indices]

    if divider.size != 0:
        n_divider = np.array(normalize_lanes(x, y, divider, heading))
        n_divider = n_divider[:, :, [1, 0]]
        n_divider[:,:,0] = -n_divider[:,:,0]
        plot_points_with_laplace_variances(n_divider[:,:,0], n_divider[:,:,1], divider_betas[:,:,0], divider_betas[:,:,1], 'orange', value['sample_token'], ax, heading)
    if ped_crossing.size != 0:
        n_ped_crossing = np.array(normalize_lanes(x, y, ped_crossing, heading))
        n_ped_crossing = n_ped_crossing[:, :, [1, 0]]
        n_ped_crossing[:,:,0] = -n_ped_crossing[:,:,0]
        plot_points_with_laplace_variances(n_ped_crossing[:,:,0], n_ped_crossing[:,:,1], ped_crossing_betas[:,:,0], ped_crossing_betas[:,:,1], 'blue', value['sample_token'], ax, heading)
    if boundary.size != 0:
        n_boundary = np.array(normalize_lanes(x, y, boundary, heading))
        n_boundary = n_boundary[:, :, [1, 0]]
        n_boundary[:,:,0] = -n_boundary[:,:,0]
        plot_points_with_laplace_variances(n_boundary[:,:,0], n_boundary[:,:,1], boundary_betas[:,:,0], boundary_betas[:,:,1], 'green', value['sample_token'], ax, heading)
    
    plt.scatter(0, 0, s=100, c='red', marker="*")

    n_ego_hist = normalize_traj(x, y, value['ego_hist'], heading)
    n_ego_fut =  normalize_traj(x, y, value['ego_fut'], heading)
    n_ego_hist[:,1] = -n_ego_hist[:,1]
    n_ego_fut[:,1] = -n_ego_fut[:,1]
    plt.plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='blue')
    plt.plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

    for i in range(6):
        n_predict_fut = normalize_traj(x, y, value['predict_fut'][i], heading)
        n_predict_fut[:,1] = -n_predict_fut[:,1]
        # breakpoint()
        plt.plot(n_predict_fut[:, 1], n_predict_fut[:, 0], c='orange')

        beta_x = value['predict_fut'][i][:, 2]
        beta_y = value['predict_fut'][i][:, 3]

        # Calculate the variance from the beta values
        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
            
        # Draw an axis-aligned ellipse around each point
        # for j in range(len(beta_x)):
        #     # Using variance to set the width and height of the ellipse
        #     ellipse = Ellipse((n_predict_fut[j][1], n_predict_fut[j][0]), width=np.sqrt(var_x[j])*2, height=np.sqrt(var_y[j])*2,
        #                     edgecolor='red', fc='None', lw=2)
        #     ax.add_patch(ellipse)

    plt.show()
    # plt.savefig('/home/guxunjia/project/HiVT_modified/plots/stream_std/' + value['sample_token'] + '.png', bbox_inches='tight', format='png',dpi=1200)
    # plt.savefig('/home/guxunjia/project/HiVT_modified/plots/stream_std/' + value['sample_token'] + '.pdf', bbox_inches='tight', format='pdf',dpi=1200)
    plt.close(fig)

