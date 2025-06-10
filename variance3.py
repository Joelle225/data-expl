import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

KEYPOINT_MAP = {
    0: 'head', 1: 'nose', 2: 'neck', 3: 'rShoulder', 4: 'rElbow',
    5: 'rWrist', 6: 'lShoulder', 7: 'lElbow', 8: 'lWrist',
    9: 'rHip', 10: 'rKnee', 11: 'rAnkle', 12: 'lHip',
    13: 'lKnee', 14: 'lAnkle', 15: 'rFoot', 16: 'lFoot'
}

def calc_variance(window):
    """
    Args:
        window: numpy array of shape [window_size, 17, 2]
    Returns:
        numpy array of shape [17, 2]: variance for each keypoint (horizontal, vertical)
    """
    # window: [window_size, 17, 2]
    # Compute variance along the time axis (axis=0) for x and y separately
    # Result: [17, 2] (keypoints, [var_x, var_y])
    window_x = window[:, :, 0]
    window_y = window[:, :, 1]
    if np.isnan(window_x).any():
        window_x = np.nan_to_num(window_x, nan=np.nanmean(window_x)) # If any are nan, set to mean of the current window

    if np.isnan(window_y).any():
        window_y = np.nan_to_num(window_y, nan=np.nanmean(window_y)) # If any are nan, set to mean of the current window

    cleaned_window = np.stack([window_x, window_y], axis=2)  # shape: [window_size, 17, 2]
    return np.var(cleaned_window, axis=0)

# Each entry is a dict: 
#   X: [T,17,2], Y: [T], meta: identifiers + frame list
#
# General shape of each entry -- dataset list where each element is:
# {
#   'X': torch.Tensor of shape [T,17,2],
#   'Y': torch.Tensor of shape [T],
#   'meta': {
#     'video':  ...,
#     'segment': ...,
#     'camera':  ...,
#     'participant': ...,
#     'frames': [list of ints]
#   }
# }
try:
    sequence_dataset_pt = torch.load("./drinking_sequence_dataset.pth")
except FileNotFoundError:
    print("Error: The file './drinking_sequence_dataset.pth' was not found.")
    print("Please ensure the dataset file is in the correct directory.")
    exit()

window_size = 30

pos_samples = []
neg_samples = []

for seq in sequence_dataset_pt:
    X_full_seq = seq['X']   # [T, 17, 2]
    Y_full_seq = seq['Y']   # [T]
    meta = seq['meta']
    T = X_full_seq.shape[0]

    for start in range(0, T - window_size + 1, window_size):
        end = start + window_size
        
        # Get the raw window data for the selected keypoints
        # X_window_raw shape: [window_size, num_selected_keypoints, 2]
        X_window_raw = X_full_seq[start:end] #[:, self.select_keypoints, :]
        
        Y_win_labels = Y_full_seq[start:end]
        label = torch.any(Y_win_labels > 0).float() # TODO: maybe adjust when the label is one, e.g. >10% drinking frames?
        
        label_np = label.item() # For scikit-learn

        data_x = X_window_raw.numpy()
        data_y = label_np

        if label_np == 1.0: # Use label_np for consistent logic
            pos_samples.append(calc_variance(data_x))
        else:
            neg_samples.append(calc_variance(data_x))

# Here we now have a sequence of variances on the x and y coordinates per keypoint, split between the positive and negative class, next we draw the results:
# Convert lists to numpy arrays for easier indexing
pos_samples_np = np.array(pos_samples)  # shape: [num_pos_samples, 17, 2]
neg_samples_np = np.array(neg_samples)  # shape: [num_neg_samples, 17, 2]

fig, axes = plt.subplots(3, 6, figsize=(18, 9))  # 17 plots, so 3x6 grid (last one empty)
axes = axes.flatten()

for k in range(17):
    ax = axes[k]
    # Horizontal variance
    pos_var_x = pos_samples_np[:, k, 0]
    neg_var_x = neg_samples_np[:, k, 0]
    # Vertical variance
    pos_var_y = pos_samples_np[:, k, 1]
    neg_var_y = neg_samples_np[:, k, 1]

    # Plot as boxplots for clarity
    ax.boxplot([pos_var_x, neg_var_x, pos_var_y, neg_var_y],
               tick_labels=['Drink X', 'Neg X', 'Drink Y', 'Neg Y'],
               patch_artist=True)
    ax.set_title(f'{KEYPOINT_MAP.get(k)}')
    ax.set_yscale('log')  # Variances can be skewed; log scale helps

# Hide the last subplot if grid > 17
for i in range(17, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle('Variance Distributions per Keypoint (Pos/Neg, X/Y)', y=1.02)
plt.savefig('variance_per_keypoint.png', bbox_inches='tight')
plt.close()

# Helpers #

def has_none(*arrays):
    return any(arr is None or np.any(pd.isnull(arr)) for arr in arrays)

def single_distance(j: np.ndarray) -> np.ndarray:
    res = np.abs(np.diff(j))
    return res

def compute_distance(j1: np.ndarray, j2: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance between two joint trajectories."""
    diff = map(lambda x: np.linalg.norm(x, axis=0), j1-j2)

    return np.array(list(diff))

def f1_head_horizontal(head: np.ndarray, nose: np.ndarray) -> float:
    if has_none(head, nose): return 0.0
    head_x = np.array(list(map(lambda x: x[0], head)))
    nose_x = np.array(list(map(lambda x: x[0], nose)))
    var_head = np.var(single_distance(head_x))
    var_nose = np.var(single_distance(nose_x))
    return (var_head + var_nose)*0.5

def f2_head_vertical(head: np.ndarray, nose: np.ndarray) -> float:
    if has_none(head, nose): return 0.0
    head_y = np.array(list(map(lambda x: x[1], head)))
    nose_y = np.array(list(map(lambda x: x[1], nose)))
    var_head = np.var(single_distance(head_y))
    var_nose = np.var(single_distance(nose_y))
    return (var_head + var_nose)*0.5
