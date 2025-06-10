import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

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

KEYPOINT_MAP = {
    0: 'head', 1: 'nose', 2: 'neck', 3: 'rShoulder', 4: 'rElbow',
    5: 'rWrist', 6: 'lShoulder', 7: 'lElbow', 8: 'lWrist',
    9: 'rHip', 10: 'rKnee', 11: 'rAnkle', 12: 'lHip',
    13: 'lKnee', 14: 'lAnkle', 15: 'rFoot', 16: 'lFoot'
}

for KEYPOINT_OF_INTEREST_IDX in range(len(KEYPOINT_MAP)):
    keypoint_name = KEYPOINT_MAP.get(KEYPOINT_OF_INTEREST_IDX, f'Keypoint {KEYPOINT_OF_INTEREST_IDX}')

    # --- 1. Calculate Features and Flatten Data ---
    # We will create a list of records for each frame, then convert it to a DataFrame.
    records = []
    for sequence in sequence_dataset_pt:
        # Convert tensors to numpy arrays for easier processing
        X = sequence['X'].numpy()
        Y = sequence['Y'].numpy()

        drinking_seqs = []
        non_drinking_seqs = []

        # TODO: find the boundaries where we switch from a sequence of 0's to 1's and vice versa, allowing for some frames of tolerance
        # --- Find boundaries with tolerance ---
        tolerance = 3  # frames of tolerance for switching
        changes = np.where(np.diff(Y) != 0)[0] + 1
        # Group close changes together
        boundaries = [0]
        for idx in changes:
            if idx - boundaries[-1] > tolerance:
                boundaries.append(idx)
        boundaries.append(len(Y))

        # TODO: extract sequences of 'drinking' and 'non-drinking' from X based on these boundries
        # --- Extract drinking/non-drinking sequences ---
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            label = Y[start]
            seq_X = X[start:end]
            if label == 1:
                drinking_seqs.append(seq_X)
            else:
                non_drinking_seqs.append(seq_X)
        
        # TODO: calculate the horizontal variance and vertical variance (remember each of the 17 nodes has an x and a y coordinate)
        # --- Calculate variances for each keypoint ---
        # For all drinking and non-drinking segments, concatenate along time
        if drinking_seqs:
            drinking_concat = np.concatenate(drinking_seqs, axis=0)  # [T', 17, 2]
            drinking_var_x = np.var(drinking_concat[:, :, 0], axis=0)  # [17]
            drinking_var_y = np.var(drinking_concat[:, :, 1], axis=0)  # [17]
        else:
            drinking_var_x = np.zeros(17)
            drinking_var_y = np.zeros(17)
        if non_drinking_seqs:
            non_drinking_concat = np.concatenate(non_drinking_seqs, axis=0)
            non_drinking_var_x = np.var(non_drinking_concat[:, :, 0], axis=0)
            non_drinking_var_y = np.var(non_drinking_concat[:, :, 1], axis=0)
        else:
            non_drinking_var_x = np.zeros(17)
            non_drinking_var_y = np.zeros(17)

        # Store for plotting
        records.append({
            'drinking_var_x': drinking_var_x,
            'drinking_var_y': drinking_var_y,
            'non_drinking_var_x': non_drinking_var_x,
            'non_drinking_var_y': non_drinking_var_y,
        })

        # TODO: plot these variances in bar graphs for each of the 17 nodes 

        # --- Plot variances in bar graphs for each of the 17 nodes ---
    # Average across all sequences
    drinking_var_x_avg = np.mean([r['drinking_var_x'] for r in records], axis=0)
    drinking_var_y_avg = np.mean([r['drinking_var_y'] for r in records], axis=0)
    non_drinking_var_x_avg = np.mean([r['non_drinking_var_x'] for r in records], axis=0)
    non_drinking_var_y_avg = np.mean([r['non_drinking_var_y'] for r in records], axis=0)

    keypoints = [KEYPOINT_MAP[i] for i in range(17)]
    x = np.arange(17)

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, drinking_var_x_avg, width=0.2, label='Drinking X')
    plt.bar(x, drinking_var_y_avg, width=0.2, label='Drinking Y')
    plt.bar(x + 0.2, non_drinking_var_x_avg, width=0.2, label='Non-Drinking X')
    plt.bar(x + 0.4, non_drinking_var_y_avg, width=0.2, label='Non-Drinking Y')
    plt.xticks(x, keypoints)
    plt.xlabel('Keypoint')
    plt.ylabel('Variance')
    plt.title(f'Variance for {keypoint_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'variance_{keypoint_name}.png')

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
