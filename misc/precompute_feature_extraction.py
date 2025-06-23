from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

window_size = 180
step_size = 5

# Before running this file, dataset[] will look as follows:
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

# After running this file, extr_feature_set[] will look as follows:
# Each entry is a dict: 
#   X: [M,N], Y: [M], meta: identifiers + frame list
# Here M is the amount of new train/test items and depends on the window size and step size used

# TODO normalize?

extr_feature_set = []

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

def analyze_keypoint_movement(sequence, hand, nose_idx=1):
    """
    Calculates movement statistics for a keypoint relative to the nose (default index 0).
    """
    keypoint_coords = sequence[:, hand, :]
    nose_coords = sequence[:, nose_idx, :]
    # Calculate relative coordinates
    rel_coords = keypoint_coords - nose_coords

    # Check for sufficient data
    if rel_coords.shape[0] < 2:
        return {'mean_speed': 0, 'max_speed': 0, 'std_speed': 0, 'total_displacement': np.array([0,0])}

    # Calculate displacements between consecutive frames (velocity vectors)
    displacements = np.diff(rel_coords, axis=0)
    # Calculate speed (magnitude of velocity) for each frame transition
    speeds = np.linalg.norm(displacements, axis=1)
    # Calculate total displacement from the start to the end of the window
    start_pos = rel_coords[0]
    end_pos = rel_coords[-1]
    total_displacement_vector = end_pos - start_pos

    movement_stats = {
        'mean_speed': np.nanmean(speeds),
        'max_speed': np.nanmax(speeds),
        'std_speed': np.nanstd(speeds),
        'total_displacement': total_displacement_vector
    }
    return movement_stats

def calculate_angle(p1, p2, p3):
    """Calculates the angle at point p2 formed by lines p1-p2 and p3-p2."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0: return np.nan # Avoid division by zero
    cosine_angle = dot_product / norm_product
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance_over_time(sequence, point1_idx, point2_idx):
        """
        Calculates the Euclidean distance between two keypoints for each frame in a sequence.

        Args:
            sequence (np.ndarray): The input sequence of shape [frames, keypoints, 2].
            point1_idx (int): The index of the first keypoint.
            point2_idx (int): The index of the second keypoint.

        Returns:
            np.ndarray: A 1D array of distances for each frame.
        """
        # Get the coordinate time series for both points
        point1_coords = sequence[:, point1_idx, :]
        point2_coords = sequence[:, point2_idx, :]
        
        # Calculate the Euclidean distance for each frame (row-wise)
        # The result is a 1D array of shape [frames]
        return np.linalg.norm(point1_coords - point2_coords, axis=1)

def hand_to_mouth_min(window, hand):
    distances = calculate_distance_over_time(window, hand, 1) # uses nose to approximate mouth
    return np.nanmin(distances)

def hand_to_mouth_max(window, hand):
    distances = calculate_distance_over_time(window, hand, 1)
    return np.nanmax(distances)

# Extracts N features
# Takes the x data (form [T,17,2]) where T is num frames
def extract_features(window):
    # return window_data_np.flatten() # for testing 
    # window_data_np: [window_size, num_selected_keypoints, 2] (e.g., [45, 5, 2])
    # Output: 1D NumPy array of features

    features = []

    features.append(hand_to_mouth_min(window, 5))
    features.append(hand_to_mouth_min(window, 8))
    features.append(hand_to_mouth_max(window, 5))
    features.append(hand_to_mouth_max(window, 8))

    lhandspeeds = analyze_keypoint_movement(window, 8)
    rhandspeeds = analyze_keypoint_movement(window, 5)

    features.append(lhandspeeds['mean_speed'])
    features.append(lhandspeeds['total_displacement'][0])
    features.append(lhandspeeds['total_displacement'][1])
    features.append(rhandspeeds['mean_speed'])
    features.append(rhandspeeds['total_displacement'][0])
    features.append(rhandspeeds['total_displacement'][1])

    variances = calc_variance(window) # 2, 3, 6, 7, 8, 9, 10, 12, 13, 15, 16

    features.extend(variances.flatten())

    # TODO: angles: rShoulder-rElbow-rWrist and lShoulder-lElbow-lWrist 
    # features.append(())

    right_arm_angles = []
    left_arm_angles = []
    
    # Calculate angle for each frame in the window
    for frame_coords in window: # frame_coords is [17, 2]
        # Right Arm: rShoulder (3) - rElbow (4) - rWrist (5)
        p1_r, p2_r, p3_r = frame_coords[3], frame_coords[4], frame_coords[5]
        right_angle = calculate_angle(p1_r, p2_r, p3_r)
        if not np.isnan(right_angle):
            right_arm_angles.append(right_angle)

        # Left Arm: lShoulder (6) - lElbow (7) - lWrist (8)
        p1_l, p2_l, p3_l = frame_coords[6], frame_coords[7], frame_coords[8]
        left_angle = calculate_angle(p1_l, p2_l, p3_l)
        if not np.isnan(left_angle):
            left_arm_angles.append(left_angle)

    # Add summary statistics of the angles as features
    # Use np.nanmean etc. in case some frames had missing keypoints
    features.append(np.mean(right_arm_angles) if right_arm_angles else 0)
    features.append(np.std(right_arm_angles) if right_arm_angles else 0)
    features.append(np.min(right_arm_angles) if right_arm_angles else 0)
    features.append(np.max(right_arm_angles) if right_arm_angles else 0)
    
    features.append(np.mean(left_arm_angles) if left_arm_angles else 0)
    features.append(np.std(left_arm_angles) if left_arm_angles else 0)
    features.append(np.min(left_arm_angles) if left_arm_angles else 0)
    features.append(np.max(left_arm_angles) if left_arm_angles else 0)
    
    return np.array(features, dtype=np.float32)

dataset = torch.load("./drinking_sequence_dataset.pth")

for entry in tqdm(dataset, desc="Extracting features..."):
    X = entry['X']

    # Get the data and metadata from the current entry
    Y = entry['Y']
    meta = entry['meta']
    T = X.shape[0]

    # Iterate through the sequence using a sliding window
    for start_idx in range(0, T - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Slice the window from the data tensors
        window_X_tensor = X[start_idx:end_idx]
        window_Y_labels = Y[start_idx:end_idx]
        
        labelp = (window_Y_labels > 0).float().mean().item()
        label = 1.0 if labelp >= 0.5 else 0.0

        if label == 0.0 and (window_Y_labels > 0).any():
            continue
        
        # 2. Extract features from the window
        # Convert to NumPy and pass to the helper function
        window_X_np = window_X_tensor.numpy()
        features = extract_features(window_X_np)
        
        # 3. Store the results in the defaultdict
        # This appends each result to the appropriate list within the dictionary
        # extr_feature_set['X'].append(features)
        # extr_feature_set['Y'].append(label)
        
        # Create and store metadata for this specific window
        window_meta = {
            **meta,
            'start_frame_idx': start_idx,
            'end_frame_idx': end_idx - 1,
            'original_frames_list': meta['frames'][start_idx:end_idx]
        }

        final = {
            'X': features,
            'Y': label,
            'meta':window_meta
        }

        extr_feature_set.append(final)

output_path = Path("./extracted_drinking_sequence_dataset_features.pth")
torch.save(extr_feature_set, output_path)
print(f"Dataset saved to {output_path.resolve()}")
