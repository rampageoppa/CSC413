import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from PIL import Image
from tqdm import tqdm

def plot_points_with_ellipsoids(x, y, sigma_x, sigma_y, rho, labels):
    colors_plt = ['orange', 'b', 'g']
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
    colors_plt = ['orange', 'b', 'g']
    # fig, ax = plt.subplots(figsize=(6, 12))
    fig, ax = plt.subplots(figsize=(2, 4))
    plt.axis('off')
    car_img = Image.open('/home/guxunjia/project/final_plots/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    for i in range(len(x)):
        plt.plot(x[i], y[i], color=colors_plt[labels[i]], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x[i], y[i], color=colors_plt[labels[i]], s=2, alpha=0.8, zorder=-1)

        # Calculate the variance from the beta values
        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        # breakpoint()
        
        # Draw an axis-aligned ellipse around each point
        for j in range(len(x[i])):
            # Using variance to set the width and height of the ellipse
            ellipse = Ellipse((x[i][j], y[i][j]), width=np.sqrt(var_x[i][j])*2, height=np.sqrt(var_y[i][j])*2,
                              fc=colors_plt[labels[i]], lw=0.5, alpha=0.5) #alpha=2, edgecolor='red'
            ax.add_patch(ellipse)
        # ax.annotate(f"{scores[i]:.2f}", (x[i][0], y[i][0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)

    # plt.show()
    plt.savefig('/home/guxunjia/project/final_plots/maptr/' + sample_idx + '.png', bbox_inches='tight', format='png', dpi=1200)
    plt.close(fig)


with open('/home/guxunjia/project/MapTR_modified/results_epoch19_mini_train.pickle', 'rb') as handle:
    data = pickle.load(handle)


for i in tqdm(range(0, len(data))):
    pts = data[i]['pts_bbox']['pts_3d']
    betas = data[i]['pts_bbox']['betas_3d']
    scores = data[i]['pts_bbox']['scores_3d']
    labels = data[i]['pts_bbox']['labels_3d']
    sample_idx = data[i]['pts_bbox']['sample_idx']
    indices = np.where(scores > 0.4)[0]
    # breakpoint()
    x = pts[indices, :, 0]  # x values
    # breakpoint()
    y = pts[indices, :, 1]  # y values
    beta_x = betas[indices, :, 0]
    beta_y = betas[indices, :, 1]

    plot_points_with_laplace_variances(x, y, beta_x, beta_y, labels[indices], scores[indices], sample_idx)


