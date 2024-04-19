import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

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
        ax.plot(x[i], y[i], color=color, linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=color, s=2, alpha=0.8, zorder=-1)

        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        
        for j in range(len(x[i])):
            ellipse = Ellipse((x[i][j], y[i][j]), width=np.sqrt(var_x[i][j])*2, height=np.sqrt(var_y[i][j])*2,
                              fc=color, lw=0.5, alpha=0.3) #alpha=2, edgecolor='red'
            ax.add_patch(ellipse)
        # ax.annotate(f"{scores[i]:.2f}", (x[i][0], y[i][0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)

def plot_points_with_laplace_variances_bev(x, y, beta_x, beta_y, color, sample_idx, ax, heading):
    for i in range(len(x)):
        ax.plot(x[i], y[i], color=color, linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=color, s=2, alpha=0.8, zorder=-1)

        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        
        for j in range(len(x[i])):
            ellipse = Ellipse((x[i][j], y[i][j]), width=0, height=0,
                              fc=color, lw=0.5, alpha=0.5) #alpha=2, edgecolor='red'
            ax.add_patch(ellipse)
        # ax.annotate(f"{scores[i]:.2f}", (x[i][0], y[i][0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)

def visualize_bev(bev_embed):
    feature_embedding = bev_embed
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

    return normalized_embedding

    # Reducing the last dimension to 3 using PCA
    # pca = PCA(n_components=256)    # 256
    # reduced_data = np.zeros((20000, 256))    # 256

    # reduced_data = pca.fit_transform(bev_embed)
    # # Applying DBSCAN clustering on the reduced data for the first batch
    # # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # # clusters = dbscan.fit_predict(reduced_data)

    # kmeans = KMeans(n_clusters=12, n_init=10)    # 12
    # clusters = kmeans.fit_predict(reduced_data)

    # # Reshaping clusters to visualize as an image (e.g., 200 x 100 grid)
    # cluster_image = clusters.reshape(200, 100)
    # # breakpoint()

    # # Displaying the cluster image
    # plt.imshow(cluster_image, cmap='viridis')
    # plt.colorbar(label='Cluster ID')
    # plt.axis('off')
    # plt.show()

# folder_path = "/home/guxunjia/project/HiVT_data/stream/mini_val/data/"
# folder_path = "/home/guxunjia/project/HiVT_data/maptr/full_val/data/"
folder_path = "/home/guxunjia/project/HiVT_data/maptr_bev/val/data/"
data = {}

for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith('.pkl'):
        scene_id = filename.replace('scene-', '').replace('.pkl', '')

        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'rb') as file:
            data_content = pickle.load(file)
        bev_ = visualize_bev(data_content['predicted_map']['bev_features'].numpy())

        del data_content['predicted_map']['bev_features']
        data_content['predicted_map']['bev_features'] = bev_
        # breakpoint()
        data[int(scene_id)] = data_content

with open('/home/guxunjia/project/HiVT_modified/bev_results/mini/maptr_al.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/HiVT_modified/bev_results/full/stream_al.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/DenseTNT_modified/bev_results/mini/stream_al.pkl', 'rb') as handle:
    predict_data = pickle.load(handle)

with open('/home/guxunjia/project/HiVT_modified/bev_results/mini/maptr_al_unc.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/HiVT_modified/bev_results/full/stream_al_unc.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/DenseTNT_modified/bev_results/mini/stream_al_unc.pkl', 'rb') as handle:
    predict_data_std = pickle.load(handle)

with open('/home/guxunjia/project/HiVT_modified/bev_results/mini/maptr_ab.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/HiVT_modified/bev_results/full/stream_ab.pkl', 'rb') as handle:
# with open('/home/guxunjia/project/DenseTNT_modified/bev_results/mini/stream_ab.pkl', 'rb') as handle:
    predict_data_bev = pickle.load(handle)

with open('/home/guxunjia/Desktop/VAD/test/result.pkl', 'rb') as handle:
    boex_gt = pickle.load(handle)
breakpoint()
for key, value in predict_data.items():
    try:
        data[int(key)]['predict_fut'] = value
        data[int(key)]['predict_fut_std'] = predict_data_std[key]
        data[int(key)]['predict_fut_bev'] = predict_data_bev[key]

        # data[int(key)]['predict_fut'] = value.reshape(1, 6, 30, 2)
        # data[int(key)]['predict_fut_std'] = predict_data_std[key].reshape(1, 6, 30, 2)
        # data[int(key)]['predict_fut_bev'] = predict_data_bev[key].reshape(1, 6, 30, 2)
    except:
        breakpoint()

for scene_index, (key, value) in enumerate(tqdm(data.items())):
    if scene_index < 2066: #1279
        pass
    else:
        breakpoint()
    x, y = value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item()
    heading = value['ego_heading'].item()
    # fig, ax = plt.subplots(figsize=(2, 4))
    fig, ax = plt.subplots(1, 3, figsize=(10, 30))
    # fig, ax = plt.subplots(1, 2, figsize=(4, 4))  # 1 row, 2 columns
    # plt.axis('off')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].set_title("No Unc")
    ax[1].set_title("Unc")
    ax[2].set_title("BEV")
    ax[2].imshow(value['predicted_map']['bev_features'], cmap='gray', alpha=0.3, extent=[-15, 15, -30, 30], aspect='auto', zorder=-2)

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
        plot_points_with_laplace_variances_bev(n_divider[:,:,0], n_divider[:,:,1], divider_betas[:,:,0], divider_betas[:,:,1], 'orange', value['sample_token'], ax[0], heading)
        plot_points_with_laplace_variances(n_divider[:,:,0], n_divider[:,:,1], divider_betas[:,:,0], divider_betas[:,:,1], 'orange', value['sample_token'], ax[1], heading)
        plot_points_with_laplace_variances_bev(n_divider[:,:,0], n_divider[:,:,1], divider_betas[:,:,0], divider_betas[:,:,1], 'orange', value['sample_token'], ax[2], heading)
    if ped_crossing.size != 0:
        n_ped_crossing = np.array(normalize_lanes(x, y, ped_crossing, heading))
        n_ped_crossing = n_ped_crossing[:, :, [1, 0]]
        n_ped_crossing[:,:,0] = -n_ped_crossing[:,:,0]
        plot_points_with_laplace_variances_bev(n_ped_crossing[:,:,0], n_ped_crossing[:,:,1], ped_crossing_betas[:,:,0], ped_crossing_betas[:,:,1], 'blue', value['sample_token'], ax[0], heading)
        plot_points_with_laplace_variances(n_ped_crossing[:,:,0], n_ped_crossing[:,:,1], ped_crossing_betas[:,:,0], ped_crossing_betas[:,:,1], 'blue', value['sample_token'], ax[1], heading)
        plot_points_with_laplace_variances_bev(n_ped_crossing[:,:,0], n_ped_crossing[:,:,1], ped_crossing_betas[:,:,0], ped_crossing_betas[:,:,1], 'blue', value['sample_token'], ax[2], heading)
    if boundary.size != 0:
        n_boundary = np.array(normalize_lanes(x, y, boundary, heading))
        n_boundary = n_boundary[:, :, [1, 0]]
        n_boundary[:,:,0] = -n_boundary[:,:,0]
        plot_points_with_laplace_variances_bev(n_boundary[:,:,0], n_boundary[:,:,1], boundary_betas[:,:,0], boundary_betas[:,:,1], 'green', value['sample_token'], ax[0], heading)
        plot_points_with_laplace_variances(n_boundary[:,:,0], n_boundary[:,:,1], boundary_betas[:,:,0], boundary_betas[:,:,1], 'green', value['sample_token'], ax[1], heading)
        plot_points_with_laplace_variances_bev(n_boundary[:,:,0], n_boundary[:,:,1], boundary_betas[:,:,0], boundary_betas[:,:,1], 'green', value['sample_token'], ax[2], heading)
    
    # plt.scatter(0, 0, s=100, c='red', marker="*")
    ax[0].scatter(0, 0, s=100, c='red', marker="*")
    ax[1].scatter(0, 0, s=100, c='red', marker="*")
    ax[2].scatter(0, 0, s=100, c='red', marker="*")

    n_ego_hist = normalize_traj(x, y, value['ego_hist'], heading)
    n_ego_fut =  normalize_traj(x, y, value['ego_fut'], heading)
    n_ego_hist[:,1] = -n_ego_hist[:,1]
    n_ego_fut[:,1] = -n_ego_fut[:,1]
    # plt.plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='blue')
    # plt.plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')
    ax[0].plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='#489ACC')
    ax[0].plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')
    ax[1].plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='#489ACC')
    ax[1].plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')
    ax[2].plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='#489ACC')
    ax[2].plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

    origin = torch.tensor(value['ego_hist'][-1], dtype=torch.float)
    av_heading_vector = origin - torch.tensor(value['ego_hist'][-2], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])
    agent_indices_original = np.where(value['agent_type'] == 1)[0]
    keep_indices = []
    flag = False
    for index in agent_indices_original:
        # n_point = normalize_point(data['ego_pos'][0, 0].item(), data['ego_pos'][0, 1].item(), data['agent_hist'][index, -1, :2],  data['ego_heading'].item())
        n_point = torch.matmul(torch.tensor([value['agent_hist'][index, -1, :2]]) - origin, rotate_mat)
        if -30 < n_point[0][0].item() < 30 and -15 < n_point[0][1] < 15:
            keep_indices.append(index)
    
    if keep_indices == []:
        pass
    else:
        for agent_index in keep_indices:
            agent_hist = value['agent_hist'][agent_index][:, :2]
            agent_fut = value['agent_fut'][agent_index]
            n_agent_hist = normalize_traj(x, y, agent_hist, heading)
            n_agent_fut =  normalize_traj(x, y, agent_fut, heading)
            n_agent_hist[:,1] = -n_agent_hist[:,1]
            n_agent_fut[:,1] = -n_agent_fut[:,1]
            # plt.plot(n_agent_hist[:, 1], n_agent_hist[:, 0], c='blue')
            # plt.plot(n_agent_fut[:, 1], n_agent_fut[:, 0], c='red')
            ax[0].plot(n_agent_hist[:, 1], n_agent_hist[:, 0], c='#489ACC')
            ax[0].plot(n_agent_fut[:, 1], n_agent_fut[:, 0], c='red')
            ax[1].plot(n_agent_hist[:, 1], n_agent_hist[:, 0], c='#489ACC')
            ax[1].plot(n_agent_fut[:, 1], n_agent_fut[:, 0], c='red')
            ax[2].plot(n_agent_hist[:, 1], n_agent_hist[:, 0], c='#489ACC')
            ax[2].plot(n_agent_fut[:, 1], n_agent_fut[:, 0], c='red')

    for i in range(len(value['predict_fut'])):
        agent_trj = value['predict_fut'][i]

        for j in range(6):
            # breakpoint()
            n_predict_fut = normalize_traj(x, y, agent_trj[j], heading)
            n_predict_fut[:,1] = -n_predict_fut[:,1]
            first_x = n_predict_fut[0][0]
            first_y = n_predict_fut[0][1]

            if not(-30 < first_x < 30 and -15 < first_y < 15):
                break
            ax[0].plot(n_predict_fut[:, 1], n_predict_fut[:, 0], c='pink')

            # beta_x = value['predict_fut'][i][j][:, 2]
            # beta_y = value['predict_fut'][i][j][:, 3]

            # # Calculate the variance from the beta values
            # var_x = 2 * beta_x ** 2
            # var_y = 2 * beta_y ** 2
                
            # Draw an axis-aligned ellipse around each point
            # for j in range(len(beta_x)):
            #     # Using variance to set the width and height of the ellipse
            #     ellipse = Ellipse((n_predict_fut[j][1], n_predict_fut[j][0]), width=np.sqrt(var_x[j])*2, height=np.sqrt(var_y[j])*2,
            #                     edgecolor='red', fc='None', lw=2)
            #     ax.add_patch(ellipse)

    for i in range(len(value['predict_fut_std'])):
        agent_trj = value['predict_fut_std'][i]

        for j in range(6):

            n_predict_fut = normalize_traj(x, y, agent_trj[j], heading)
            n_predict_fut[:,1] = -n_predict_fut[:,1]
            first_x = n_predict_fut[0][0]
            first_y = n_predict_fut[0][1]

            if not(-30 < first_x < 30 and -15 < first_y < 15):
                break
            ax[1].plot(n_predict_fut[:, 1], n_predict_fut[:, 0], c='pink')


    for i in range(len(value['predict_fut_bev'])):
        agent_trj = value['predict_fut_bev'][i]

        for j in range(6):

            n_predict_fut = normalize_traj(x, y, agent_trj[j], heading)
            n_predict_fut[:,1] = -n_predict_fut[:,1]
            first_x = n_predict_fut[0][0]
            first_y = n_predict_fut[0][1]

            if not(-30 < first_x < 30 and -15 < first_y < 15):
                break
            ax[2].plot(n_predict_fut[:, 1], n_predict_fut[:, 0], c='pink')

    # if value['sample_token'] == '61a7bd24f88a46c2963280d8b13ac675':

    plt.show()
    # plt.savefig('/home/guxunjia/project/HiVT_modified/bev_plots/full/maptr/' + value['sample_token'] + '.png', bbox_inches='tight', format='png',dpi=100) #dpi=1200
    # plt.savefig('/home/guxunjia/project/HiVT_modified/plots/stream_std/' + value['sample_token'] + '.pdf', bbox_inches='tight', format='pdf',dpi=1200)
    plt.close(fig)

def check_in_range(traj):
    for i in range(traj.shape[0]):      # Loop over first dimension
        x, y = traj

