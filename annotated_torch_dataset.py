import torch
import numpy as np
from torch.utils.data import Dataset
import random
from tqdm import tqdm

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
        neg_to_pos_ratio=3,
        balance=True,
        # jitter_max=5, # Jitter is complex with pre-extracted features, disable for RF
        reverse_positives=True, # Reversing raw data before feature extraction is valid
        seed=42,
        is_for_sklearn=False # New flag
    ):
        self.window_size = window_size
        self.stride = stride
        # self.jitter_max = jitter_max # Disabling for RF simplicity
        self.reverse_positives = reverse_positives
        self.balance = balance
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.samples = []
        self.is_for_sklearn = is_for_sklearn

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

        # features.append(())
            
        return np.array(features)

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
            return X_data.nan_to_num(nan=1.0), torch.tensor(Y_data).nan_to_num(nan=1.0)

