import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from nuscenes.nuscenes import NuScenes

def plot_points_with_ellipsoids(x, y, sigma_x, sigma_y, rho, labels):
    colors_plt = ['b', 'orange', 'g']
    fig, ax = plt.subplots(figsize=(6,12))

    for i in range(len(x)):
        plt.plot(x[i], y[i], color=colors_plt[labels[i]], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=colors_plt[labels[i]])
        # Draw an ellipse around each point
        for j in range(len(x[i])):
            # Calculate angle of rotation for the ellipse
            angle = np.arctan(rho[j]) * 180 / np.pi
            ellipse = Ellipse((x[i][j], y[i][j]), width=sigma_x[j]*2, height=sigma_y[j]*2,
                            angle=angle, edgecolor='red', fc='None', lw=2)
            ax.add_patch(ellipse)

    plt.show()

def plot_points_with_laplace_variances(x, y, beta_x, beta_y, labels, scores, sample_idx):
    colors_plt = ['b', 'orange', 'g']
    # fig, ax = plt.subplots(figsize=(6, 12))
    fig, ax = plt.subplots(figsize=(2, 4))

    plt.axis('off')

    car_img = Image.open('/home/guxunjia/project/final_plots/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    # plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

    for i in range(len(x)):
        plt.plot(x[i], y[i], color=colors_plt[labels[i]], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=colors_plt[labels[i]], s=2, alpha=0.8, zorder=-1)

        # Calculate the variance from the beta values
        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        
        # Draw an axis-aligned ellipse around each point
        for j in range(len(x[i])):
            # Using variance to set the width and height of the ellipse
            ellipse = Ellipse((x[i][j], y[i][j]), width=np.sqrt(var_x[i][j])*2*30, height=np.sqrt(var_y[i][j])*2*60,
                              fc=colors_plt[labels[i]], lw=0.5, alpha=0.5) #alpha=2, edgecolor='red'
            ax.add_patch(ellipse)
        # ax.annotate(f"{scores[i]:.2f}", (x[i][0], y[i][0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)
    plt.gca().invert_xaxis()
    # plt.show()
    plt.savefig('/home/guxunjia/project/final_plots/stream/' + sample_idx + '.png', bbox_inches='tight', format='png',dpi=1200)
    plt.close(fig)


def scale_values(values, old_min, old_max, new_min, new_max):
    return new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)

def calculate_std(x, y, beta_x, beta_y, labels, scores, sample_idx, nusc):
    var_x = 2 * beta_x ** 2
    var_y = 2 * beta_y ** 2
    std_x = np.sqrt(var_x) * 30
    std_y = np.sqrt(var_y) * 60

    uncertainties = np.sqrt(std_x**2 + std_y**2)
    max_uncertainties = np.max(uncertainties, axis=1)
    indices = np.argmax(uncertainties, axis=1)

    uncertainties = max_uncertainties

    pos = np.stack((x, y), axis=2)

    distances = np.sqrt(x[np.arange(len(x)), indices]**2 + y[np.arange(len(y)), indices]**2).flatten()

    average_uncertainty = np.mean(uncertainties)

    new_labels = labels

    # Sample token
    sample_token = sample_idx

    # Fetch the sample record
    sample_record = nusc.get('sample', sample_token)

    # Get the scene token from the sample record
    scene_token = sample_record['scene_token']

    # Fetch the scene record
    scene_record = nusc.get('scene', scene_token)

    if 'night' in scene_record['description'].lower() or 'difficult lighting' in scene_record['description'].lower():
        night_tag = np.ones_like(new_labels)
    else:
        night_tag = np.zeros_like(new_labels)

    if 'rain' in scene_record['description'].lower():
        rain_tag = np.ones_like(new_labels)
    else:
        rain_tag = np.zeros_like(new_labels)

    if 'turn' in scene_record['description'].lower():
        turn_tag = np.ones_like(new_labels)
    else:
        turn_tag = np.zeros_like(new_labels)

    if 'intersection' in scene_record['description'].lower():
        intersection_tag = np.ones_like(new_labels)
    else:
        intersection_tag = np.zeros_like(new_labels)
    # Get the scene ID
    # scene_id = scene_record['name']
    # breakpoint()
    # print(scene_record['description'])
    tags = [night_tag, rain_tag, turn_tag, intersection_tag]

    return distances, uncertainties, new_labels, tags

def get_speed(positions):
    # Time interval
    dt = 0.1

    # Calculate differences in position
    differences = np.diff(positions, axis=0)

    # Calculate velocity (change in position over time)
    velocity = differences / dt

    # If you need the magnitude of velocity (speed)
    speed = np.linalg.norm(velocity, axis=1)
    # breakpoint()
    return speed

with open('/home/guxunjia/project/pkl_files_adaptor_stream/results_full_val.pickle', 'rb') as handle:
    data = pickle.load(handle)

with open('/home/guxunjia/project/pkl_files_adaptor_stream/vec_scene_full_val.pkl', 'rb') as handle:
    motion_data = pickle.load(handle)


all_distances = np.array([])
all_uncertainties = np.array([])
all_labels = np.array([])
all_night_tags = np.array([])
all_rain_tags = np.array([])
all_turn_tags = np.array([])
all_intersection_tags = np.array([])
all_speeds = np.array([])
# Initialize the nuScenes API
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/data/nuscenes', verbose=True)

for i in tqdm(range(0, len(data))):
    pts = data[i]['vectors']
    betas = data[i]['betas']
    scores = data[i]['scores']
    labels = data[i]['labels']
    sample_idx = data[i]['token']
    indices = np.where(scores > 0.4)[0]

    x = pts[indices, :, 0]  # x values
    y = pts[indices, :, 1]  # y values

    # Example usage
    old_x_min, old_x_max = 0, 1
    new_x_min, new_x_max = -30, 30

    old_y_min, old_y_max = 0, 1
    new_y_min, new_y_max = -15, 15

    # Scale the values to the new ranges
    # Apply the scaling function to each element in the x and y arrays
    x_scaled = np.empty_like(x)
    y_scaled = np.empty_like(y)

    # Scale the x components
    x_scaled = scale_values(x, old_x_min, old_x_max, new_x_min, new_x_max)

    # Scale the y components
    y_scaled = scale_values(y, old_y_min, old_y_max, new_y_min, new_y_max)

    x = x_scaled
    y = y_scaled

    beta_x = betas[indices, :, 0]
    beta_y = betas[indices, :, 1]

    # print(sample_idx)
    # plot_points_with_laplace_variances(y, x, beta_y, beta_x, labels[indices], scores[indices], sample_idx)
    # breakpoint()
    for key, value in motion_data.items():
        if value['sample_token'] == sample_idx:
            distances, uncertainties, labels, tags = calculate_std(y, x, beta_y, beta_x, labels[indices], scores[indices], sample_idx, nusc)
            speed = get_speed(value['ego_hist'])
            speeds = speed[-1] * np.ones_like(distances)
            
            night_tags = tags[0]
            rain_tags = tags[1]
            turn_tags = tags[2]
            intersection_tags = tags[3]
            all_distances = np.concatenate((all_distances, distances))
            all_uncertainties = np.concatenate((all_uncertainties, uncertainties))
            all_labels = np.concatenate((all_labels, labels))
            all_night_tags = np.concatenate((all_night_tags, night_tags))
            all_rain_tags = np.concatenate((all_rain_tags, rain_tags))
            all_turn_tags = np.concatenate((all_turn_tags, turn_tags))
            all_intersection_tags = np.concatenate((all_intersection_tags, intersection_tags))
            all_speeds = np.concatenate((all_speeds, speeds))
            break

all_tags = {'night_tag': all_night_tags, 'rain_tag': all_rain_tags, 'turn_tag': all_turn_tags, 'intersection_tag': all_intersection_tags}
filtered_distances = all_distances
filtered_uncertainties = all_uncertainties
filtered_labels = all_labels

# Fit a linear regression line
coefficients = np.polyfit(filtered_distances, filtered_uncertainties, 1)
slope, intercept = coefficients

# Create a line based on the regression
regression_line = slope * filtered_distances + intercept

# Define a mapping from existing labels to new categories
label_mapping = {
    0: 'Pedestrian Crossing',
    1: 'Divider',
    2: 'Boundary',
}
new_labels = np.array([label_mapping[label] for label in filtered_labels])

# Define the bin edges for the x-axis
x_bins = np.arange(0, 35, 5)  # Bins every 2 meters from 0 to 20 meters
custom_palette = ["#008000", "#FFA500", "#0000FF"]

# Create a pointplot with Seaborn and bin the x-axis values
plt.figure(figsize=(12, 6))
ax = plt.gca()
sns.set_palette(custom_palette)
sns.pointplot(x=pd.cut(filtered_distances, bins=x_bins), y=filtered_uncertainties, hue=new_labels, dodge=True, ax=ax)
# sns.set(rc={'xtick.labelsize': 14, 'ytick.labelsize': 14})

ax.legend(loc='upper left')
# plt.plot(filtered_distances, regression_line, color='r', label='Linear Regression')
plt.xticks(fontsize=23)  
plt.yticks(fontsize=23)
plt.xlabel('Distances (m)', fontsize=25)
plt.ylabel('Uncertainties (m)', fontsize=25)
plt.legend(fontsize=20)
plt.xticks(rotation=45)
# plt.show()
# plt.savefig('/home/guxunjia/project/final_plots/distance/' + 'Stream' + '.png', bbox_inches='tight', format='png',dpi=1200)
# plt.savefig('/home/guxunjia/project/final_plots/distance/' + 'Stream' + '.pdf', bbox_inches='tight', format='pdf',dpi=1200)
# breakpoint()
data = {'distance': filtered_distances, 'uncertainties': filtered_uncertainties, 'labels': new_labels, 'extra_tag': all_tags, 'speed': all_speeds}
save_path = '/home/guxunjia/project/distance_data.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(data, file)


