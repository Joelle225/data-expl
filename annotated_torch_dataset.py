import torch
import numpy as np
from torch.utils.data import Dataset
import random
from tqdm import tqdm

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
    distances = calculate_distance_over_time(sequence, hand_idx, mouth_idx)
    # Use nanmax to be robust to frames where a keypoint might be missing (NaN)
    return np.nanmax(distances)

def hand_to_mouth_max(window, hand):
    pass

def max_hand_to_mouth_distance(sequence, hand_idx, mouth_idx):
    """Computes the maximum distance between the hand and mouth in a sequence."""
    distances = calculate_distance_over_time(sequence, hand_idx, mouth_idx)
    # Use nanmax to be robust to frames where a keypoint might be missing (NaN)
    return np.nanmax(distances)

def min_hand_to_mouth_distance(sequence, hand_idx, mouth_idx):
    """Computes the minimum distance between the hand and mouth in a sequence."""
    distances = calculate_distance_over_time(sequence, hand_idx, mouth_idx)
    # Use nanmin for robustness
    return np.nanmin(distances)


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
        
        # Define selected keypoints (head, shoulders, hands)
        # Indices for COCO: 0 (nose), [2 (LShoulder), 3(RShoulder)] OR [5,6], 7(LElbow), 8(RElbow), 9(LWrist), 10(RWrist)
        # Your original code used: 0, 2, 3, 7, 8. Let's assume these are Nose, LShoulder, RShoulder, LElbow, RElbow
        # This gives 5 keypoints.
        # self.select_keypoints = [0, 2, 3, 7, 8] # Ensure this matches your intended keypoints

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
        features.append((window))
            
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

