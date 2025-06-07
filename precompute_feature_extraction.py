from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

window_size = 45
step_size = 5
label_minimum_percentage = 5

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

extr_feature_set = defaultdict(list)

# Extracts N features
def extract_features(self, window_data_np):
        # return window_data_np.flatten() # for testing 
        # window_data_np: [window_size, num_selected_keypoints, 2] (e.g., [45, 5, 2])
        # Output: 1D NumPy array of features
        
        if window_data_np.size == 0: # Handle empty window data
            # Return a vector of NaNs or zeros of the expected feature length
            # For 5 keypoints, 2 coords, 6 stats: 5 * 2 * 6 = 60 features
            return np.full(len(self.select_keypoints) * 2 * 6, np.nan)

        num_frames, num_keypoints, num_dims = window_data_np.shape
        features = []

        # Reshape to [num_frames, num_keypoints * num_dims] e.g. [45, 10]
        flattened_coords = window_data_np.reshape(num_frames, -1)

        for i in range(flattened_coords.shape[1]): # Iterate over each of the 10 coordinate time series
            coord_timeseries = flattened_coords[:, i]
            if np.all(np.isnan(coord_timeseries)): # All NaNs in this timeseries
                features.extend([np.nan] * 6) # Add 6 NaN features
                continue

            features.append(np.nanmean(coord_timeseries))
            features.append(np.nanstd(coord_timeseries))
            features.append(np.nanmin(coord_timeseries))
            features.append(np.nanmax(coord_timeseries))
            features.append(np.nanmedian(coord_timeseries))
            features.append(np.nanmax(coord_timeseries) - np.nanmin(coord_timeseries)) # Range
            
        return np.array(features)

dataset = torch.load("./drinking_sequence_dataset.pth")

for entry in tqdm(dataset, desc="Extracting features..."):
    X = entry['X']
    # Define the keypoints to use, based on your SlidingWindowPoseDataset class
    select_keypoints = [0, 2, 3, 7, 8]  # e.g., Nose, Shoulders, Elbows
    num_selected_keypoints = len(select_keypoints)

    # Get the data and metadata from the current entry
    Y = entry['Y']
    meta = entry['meta']
    T = X.shape[0]

    # Select only the specified keypoints from the full sequence tensor
    X_selected = X[:, select_keypoints, :]

    # Iterate through the sequence using a sliding window
    for start_idx in range(0, T - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Slice the window from the data tensors
        window_X_tensor = X_selected[start_idx:end_idx]
        window_Y_labels = Y[start_idx:end_idx]
        
        # 1. Determine the label for the entire window
        # A window is labeled '1' if the percentage of positive frames meets the threshold
        positive_frames_count = torch.sum(window_Y_labels > 0).item()
        label = 1 if (positive_frames_count / window_size * 100) >= label_minimum_percentage else 0
        
        # 2. Extract features from the window
        # Convert to NumPy and pass to the helper function
        window_X_np = window_X_tensor.numpy()
        features = extract_features(window_X_np, num_selected_keypoints)
        
        # 3. Store the results in the defaultdict
        # This appends each result to the appropriate list within the dictionary
        extr_feature_set['X'].append(features)
        extr_feature_set['Y'].append(label)
        
        # Create and store metadata for this specific window
        window_meta = {
            **meta,
            'start_frame_idx': start_idx,
            'end_frame_idx': end_idx - 1,
            'original_frames_list': meta['frames'][start_idx:end_idx]
        }
        extr_feature_set['meta'].append(window_meta)

output_path = Path("./drinking_sequence_dataset_extracted.pth")
torch.save(extr_feature_set, output_path)
print(f"Dataset saved to {output_path.resolve()}")
