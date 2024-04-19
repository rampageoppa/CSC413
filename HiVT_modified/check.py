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

def compute_nll(data):
    nll_x = np.log(2 * data[:, 2])  # since |x-x| = 0
    nll_y = np.log(2 * data[:, 3])  # since |y-y| = 0

    total_nll = np.sum(nll_x + nll_y)
    return total_nll

def avg_std(element):
    all_std_x = []
    all_std_y = []
    all_predict_std_x = []
    all_predict_std_y = []
    for key, value in data.items():
        # beta_x = value['vec_map'][element+'_betas'][:, :, 0]
        # beta_y = value['vec_map'][element+'_betas'][:, :, 1]
        # std_x = np.sqrt(2 * beta_x ** 2)
        # std_y = np.sqrt(2 * beta_y ** 2)

        # std = np.sqrt(2*value['vec_map'][element+'_betas']**2)
        # breakpoint()
        predict_beta_x = value['predict_fut'][:, :, 2]
        predict_beta_y = value['predict_fut'][:, :, 3]
        predict_std_x = np.sqrt(2 * predict_beta_x ** 2)
        predict_std_y = np.sqrt(2 * predict_beta_y ** 2)

        # all_std_x.extend(std_x.flatten())
        # all_std_y.extend(std_y.flatten())
        all_predict_std_x.extend(predict_std_x.flatten())
        all_predict_std_y.extend(predict_std_y.flatten())

    # avg_std_x = sum(all_std_x) / len(all_std_x)
    # avg_std_y = sum(all_std_y) / len(all_std_y)
    avg_predict_std_x = sum(all_predict_std_x) / len(all_predict_std_x)
    avg_predict_std_y = sum(all_predict_std_y) / len(all_predict_std_y)
    # print("Avg_std_x_"+element+": "+str(avg_std_x))
    # print("Avg_std_y_"+element+": "+str(avg_std_y))
    print("Avg_predict_std_x: " +str(avg_predict_std_x))
    print("Avg_predict_std_y: "+ str(avg_predict_std_y))

def avg_std_xy(data):
    all_predict_std_x = []
    all_predict_std_y = []
    for key, value in data.items():
        # breakpoint()
        try:
            predict_beta_x = value['predict_fut'][:, :, 2]
            predict_beta_y = value['predict_fut'][:, :, 3]
            predict_std_x = np.sqrt(2 * predict_beta_x ** 2)
            predict_std_y = np.sqrt(2 * predict_beta_y ** 2)
        except:
            pass

        all_predict_std_x.extend(predict_std_x.flatten())
        all_predict_std_y.extend(predict_std_y.flatten())


    avg_predict_std_x = sum(all_predict_std_x) / len(all_predict_std_x)
    avg_predict_std_y = sum(all_predict_std_y) / len(all_predict_std_y)
    print("Avg_predict_std_x: " +str(avg_predict_std_x))
    print("Avg_predict_std_y: "+ str(avg_predict_std_y))


# folder_path = "/home/data/adapted_data/nuscene_gt/full_val_centerline"
folder_path = "/home/data/adapted_data/maptr_with_uncertainty/full_val_centerline"
# folder_path = "/home/guxunjia/project/HiVT_data_maptr/val/data/"
data = {}

for filename in os.listdir(folder_path):
    # breakpoint()
    if filename.endswith('.pkl'):
        scene_id = filename.replace('scene-', '').replace('.pkl', '')

        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'rb') as file:
            data_content = pickle.load(file)
        
        data[int(scene_id)] = data_content

with open('/home/guxunjia/project/HiVT_modified/results/maptr/result_pt_cent.pkl', 'rb') as handle:
    predicted_data = pickle.load(handle)

for key, value in predicted_data.items():
    try:
        data[key]['predict_fut'] = value
    except:
        breakpoint()


# avg_std('boundary')
# avg_std('divider')
# avg_std('ped_crossing')
# avg_std('centerline')

# breakpoint()
for key, value in data.items():
    # centerlines = value['vec_map']['centerlines']
    # try:
    #     left_edges = value['vec_map']['left_edges']
    # except:
    #     breakpoint()
    # right_edges = value['vec_map']['right_edges']
    centerlines = value['vec_map']['centerlines']
    try:
        left_edges = value['gt_map']['left_edges']
    except:
        breakpoint()
    right_edges = value['gt_map']['right_edges']

    n_centerlines = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), centerlines, value['ego_heading'].item())
    n_left_edges = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), left_edges, value['ego_heading'].item())
    n_right_edges = normalize_lanes(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), right_edges, value['ego_heading'].item())

    fig, ax = plt.subplots(figsize=(6, 12))

    for lane in n_centerlines:
        plt.plot(lane[:, 1], lane[:, 0], c='grey', linestyle='--')
    
    for lane in n_left_edges:
        plt.plot(lane[:, 1], lane[:, 0], c='black')

    for lane in n_right_edges:
        plt.plot(lane[:, 1], lane[:, 0], c='black')
    
    plt.scatter(0, 0, s=100, c='red', marker="*")
    # breakpoint()

    n_ego_hist = normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['ego_hist'], value['ego_heading'].item())
    n_ego_fut =  normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['ego_fut'], value['ego_heading'].item())
    plt.plot(n_ego_hist[:, 1], n_ego_hist[:, 0], c='blue')
    plt.plot(n_ego_fut[:, 1], n_ego_fut[:, 0], c='red')

    for i in range(6):
        n_predict_fut = normalize_traj(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['predict_fut'][i], value['ego_heading'].item())
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
        n_point = normalize_point(value['ego_pos'][0, 0].item(), value['ego_pos'][0, 1].item(), value['agent_hist'][i, -1, :2],  value['ego_heading'].item())
        if -30 <= n_point[0] <= 30 and -15 <= n_point[1] <= 15:
            plt.scatter(n_point[1], n_point[0], s=100, c=color[value['agent_type'][i].item()], marker="*")
        else:
            continue

    # print(value['sample_token'])
    print(value['scene_name'])

    ax.invert_xaxis()
    plt.show()
    # breakpoint()
    # plt.savefig('/home/guxunjia/project/HiVT_modified/gt_plots/' + value['scene_name'] + '.png', dpi=300)
    plt.close(fig)
    # breakpoint()

# breakpoint()