import numpy as np
import matplotlib.pyplot as plt
import transforms as trans
import os

plt.rcParams['figure.figsize'] = [10, 6]

directory = os.path.dirname(os.path.abspath(__file__))

print(directory)

source = plt.imread(directory + '\mona.jpg') / 255.

## Basic transformations to manipulate the source image.
T = np.array([[1, 0, -source.shape[1] / 2],
              [0, 1, -source.shape[0] / 2],
              [0, 0, 1]])
t = np.pi / 4
R = np.array([[np.cos(t), -np.sin(t), 0],
              [np.sin(t),  np.cos(t), 0],
              [ 0, 0, 1]])
S = np.diag([2, 2, 1])

## The warping transformation (rotation about arbitrary point).
M = np.linalg.inv(T) @ R @ S @ T

### FORWARD MAPPING ###

target = trans.forward_mapping(source, M)
plt.imshow(np.hstack([source, target]))
plt.axis("off");
plt.show()

### BACKWARD MAPPING ###

target = trans.backward_mapping(source, M)
plt.imshow(np.hstack([source, target]))
plt.axis("off");
plt.show()

### BILINEAR INTERPOLATION ### 

M = np.array([[12, 0, -2486], [0, 12, -2508], [0, 0, 1]])  # big smile
#M = np.array([[40, 0, 80], [0, 40, 80], [0, 0, 1]])  # check edge handling
target_nearest  = trans.backward_mapping(source, M)
target_bilinear = trans.backward_mapping_bilinear(source, M)
plt.imshow(np.hstack([target_nearest, target_bilinear]))
plt.axis("off");
plt.show()

### LENS UNDISTORTION ###

source = plt.imread(directory + '\window.jpg') / 255.
camera_matrix = np.array([[474.53, 0, 405.96], [0, 474.53, 217.81], [0, 0, 1]])
dist_coeffs = np.array([-0.27194, 0.11517, -0.029859])

target = trans.undistort_image_vectorised(source, camera_matrix, dist_coeffs)
plt.rcParams['figure.figsize'] = [10, 10]
plt.imshow(np.vstack([source, target]))
plt.axis("off");
plt.show()