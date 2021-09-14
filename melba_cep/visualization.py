import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import os


def calculate_offsets(predictions, targets, normalization_multiplier=1.0):
    offsets_dict = {}
    for key, values in predictions.items():
        prediction_points = values
        target_points = targets[key]
        offsets = []
        normalization = normalization_multiplier
        for i in range(len(prediction_points)):
            offsets.append((prediction_points[i].coords - target_points[i].coords) * normalization)
        offsets_dict[key] = offsets
    return offsets_dict

def plot_offsets(predictions, targets, filename, image_folder):
    offsets_dict = calculate_offsets(predictions, targets)
    imagesize = [1935, 2400]
    target_id = list(sorted(targets.keys()))[0]
    image_filename = os.path.join(image_folder, target_id + '.bmp')
    target_points = targets[target_id]
    normalization_factor = 0.1
    normalized_target_points = np.array([p.coords for p in target_points])

    f = plt.figure(frameon=False)
    f.set_size_inches(imagesize[0]/250, imagesize[1]/250)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    image = imread(image_filename)
    plt.imshow(image)
    color_list = ['#7a0724', '#e6a029', '#39a32a', '#a76542', '#064022', '#4044dc', '#8f091d', '#c74c20', '#f7d87c', '#6ec1ed', '#c2f576', '#a70a0a', '#7b402a', '#ca1818', '#f7ca40', '#2d0f84', '#4085dc', '#581414', '#ca8655', '#cc7223']

    for i in range(len(normalized_target_points)):
        current_target = normalized_target_points[i, :]
        current_offsets = np.array(list(offsets_dict.values()))[:, i, :]
        color = color_list[i]
        values = (current_target + current_offsets) / normalization_factor
        plt.plot(values[:, 0], values[:, 1], 'o', color=color, markersize=1)

    plt.plot(normalized_target_points[:, 0] / normalization_factor, normalized_target_points[:, 1] / normalization_factor, 'o', color='black', markersize=1)
    f.savefig(filename)
    plt.close('all')

def plot_sigma_error(sigmas, pes, filename):
    f = plt.figure()
    plt.scatter(sigmas, pes)
    for i, coord in enumerate(zip(sigmas, pes)):
        plt.annotate(str(i), coord)
    plt.xlabel('Sigma Product')
    plt.ylabel('Point Error (in mm)')
    f.savefig(filename)
    plt.close('all')
