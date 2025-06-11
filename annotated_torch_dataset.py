import torch
import numpy as np
from torch.utils.data import Dataset
import random
from tqdm import tqdm

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

### Class

class SlidingWindowPoseDataset(Dataset):
    def __init__(
        self,
        sequences,
        window_size=45,
        stride=1,
        neg_to_pos_ratio=4,
        balance=True,
        # jitter_max=5, # Jitter is complex with pre-extracted features, disable for RF
        reverse_positives=True, # Reversing raw data before feature extraction is valid
        seed=42,
        is_for_sklearn=False, # New flag
        min_1_label=0.30
    ):
        self.window_size = window_size
        self.stride = stride
        # self.jitter_max = jitter_max # Disabling for RF simplicity
        self.reverse_positives = reverse_positives
        self.balance = balance
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.samples = []
        self.is_for_sklearn = is_for_sklearn
        self.min_1_label=min_1_label

        random.seed(seed)
        np.random.seed(seed)

        pos_samples = []
        neg_samples = []

        for seq in tqdm(sequences, desc="Initializing Dataset"):
            X_full_seq = seq['X']   # [T, 17, 2]
            Y_full_seq = seq['Y']   # [T]
            meta = seq['meta']
            T = X_full_seq.shape[0]

            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                
                # Get the raw window data for the selected keypoints
                # X_window_raw shape: [window_size, num_selected_keypoints, 2]
                X_window_raw = X_full_seq[start:end] #[:, self.select_keypoints, :]
                
                Y_win_labels = Y_full_seq[start:end]
                label = torch.any(Y_win_labels > 0).float() # TODO: maybe adjust when the label is one, e.g. >10% drinking frames?
                # positive_fraction = (Y_win_labels > 0).float().mean().item() # uncomment this for the behavior where a portion at least needs to be drinking
                # label = 1.0 if positive_fraction >= self.min_1_label else 0.0
                # label = torch.tensor(label, dtype=torch.float32)

                label_np = label.item() # For scikit-learn

                # Extract features if for scikit-learn
                if self.is_for_sklearn:
                    # features_np = self._extract_features_for_sklearn(X_window_raw.numpy()) # Pass NumPy array
                    data_x = X_window_raw.numpy()
                    data_y = label_np
                else: # Original PyTorch tensor format
                    data_x = X_window_raw 
                    data_y = label

                sample = {
                    'X': data_x,
                    'Y': data_y,
                    'meta': {
                        **meta,
                        'start_frame': meta['frames'][start],
                        'end_frame': meta['frames'][end - 1],
                        'start_idx': start
                    }
                }

                if label_np == 1.0: # Use label_np for consistent logic
                    pos_samples.append(sample)
                    if reverse_positives:
                        X_reversed_window_raw = torch.flip(X_window_raw, dims=[0])
                        if self.is_for_sklearn:
                            # rev_features_np = self._extract_features_for_sklearn(X_reversed_window_raw.numpy())
                            rev_data_x = X_reversed_window_raw.numpy()
                        else:
                            rev_data_x = X_reversed_window_raw
                        
                        rev_sample = {
                            'X': rev_data_x,
                            'Y': data_y, # Label remains the same
                            'meta': {**sample['meta'], 'reversed': True}
                        }
                        pos_samples.append(rev_sample)
                else:
                    neg_samples.append(sample)

        if balance:
            target_n_pos = len(pos_samples)
            if target_n_pos == 0 and len(neg_samples) > 0: # No positive samples
                 print(f"Warning: No positive samples found. Using all {len(neg_samples)} negative samples (unbalanced).")
                 self.samples = neg_samples # Or decide to keep a subset
            elif target_n_pos == 0 and len(neg_samples) == 0: # No samples at all
                 print("Warning: No positive or negative samples found.")
                 self.samples = []
            else: # Positive samples exist
                keep_n_neg = min(len(neg_samples), target_n_pos * neg_to_pos_ratio)
                random.shuffle(neg_samples)
                self.samples = pos_samples + neg_samples[:keep_n_neg]
        else:
            self.samples = pos_samples + neg_samples

        if self.is_for_sklearn:
            for sample in tqdm(self.samples, desc="Extracting features..."):
                sample['X'] = self._extract_features_for_sklearn(sample['X'])

        if self.samples: # Only shuffle if there are samples
            random.shuffle(self.samples)
        else:
            print("Warning: Dataset is empty after processing and balancing.")

    # Takes the x data (form [T,17,2]) where T is num frames
    def _extract_features_for_sklearn(self, window):
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

        variances = calc_variance(window)

        # features.append((variances[0, 0] + variances[2, 0]) * 0.5) # horizontal variance of multiple keypoints (neck & nose)
        # features.append((variances[0, 1] + variances[2, 1]) * 0.5) # vertical variance of multiple keypoints

        # features.append((variances[12, 0] + variances[9, 0]) * 0.5) # horizontal variance of multiple keypoints (rhip & lhip)
        # features.append((variances[12, 1] + variances[9, 1]) * 0.5) # vertical variance of multiple keypoints

        # features.append((variances[12, 0] + variances[9, 0]) * 0.5) # horizontal variance of multiple keypoints (lwrist & rwrist)
        # features.append((variances[12, 1] + variances[9, 1]) * 0.5) # vertical variance of multiple keypoints

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

        # --- Targeted Variance Features (More effective than flatten) ---
        # variances = calc_variance(window)
        # # Get variance for hands, head, and shoulders
        # features.append(variances[5, 1]) # Vertical variance of rWrist
        # features.append(variances[8, 1]) # Vertical variance of lWrist
        # features.append(variances[0, 1]) # Vertical variance of head
        # features.append(variances[3, 1]) # Vertical variance of rShoulder
        # features.append(variances[6, 1]) # Vertical variance of lShoulder
        
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        X_data = sample['X']
        Y_data = sample['Y']

        if self.is_for_sklearn:
            # Features are already NumPy arrays, replace NaNs if any
            X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0) # RF can't handle NaNs
            return X_data, Y_data # Y_data is already a float 0.0 or 1.0
        else:
            # Original PyTorch tensor handling
            return X_data.nan_to_num(nan=0.0), torch.tensor(Y_data).nan_to_num(nan=0.0)

