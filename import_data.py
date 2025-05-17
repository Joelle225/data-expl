import torch
import numpy as np
import json
from collections import defaultdict

def pose_json_to_tensor(file):
    """
    Convert JSON data to a tensor.
    Args:
        file (str): Path to the JSON file.
    Returns:
        torch.Tensor: A tensor of shape (N, T, K, 2) where:
            N = number of participants
            T = number of frames
            K = number of keypoints
            2 = x and y coordinates of each keypoint
    """
    with open(file, "r") as f:
        data = json.load(f)
    
    skeletons = data["annotations"]["skeletons"]
    num_keypoints_K = 17  # Max number of keypoints (when indicated as less, some of them will be null values)

    # Get all unique participant IDs
    participant_ids = set()
    for frame in skeletons:
        participant_ids.update(frame.keys())
    participant_ids = sorted(participant_ids, key=lambda x: int(x)) # Now sorted by integer value

    num_participants_N = len(participant_ids)
    num_frames_T = len(skeletons)

    # Map participant ID to row index in tensor
    pid_to_index = {pid: i for i, pid in enumerate(participant_ids)}

    # Initialize pose tensor with NaNs to handle missing data
    pose_tensor = torch.full((num_participants_N, num_frames_T, num_keypoints_K, 2), float('nan'))

    # Fill the tensor
    for t, frame in enumerate(skeletons):
        for pid, pose_data in frame.items():
            keypoints = pose_data["keypoints"] # Should be the array of coordinates
            # Clean up none values TODO: Idk how to actually handle these
            keypoints = [x if x is not None else float('nan') for x in keypoints] 
            keypoints = np.array(keypoints).reshape(num_keypoints_K, 2)
            index = pid_to_index[pid]
            pose_tensor[index, t] = torch.tensor(keypoints, dtype=torch.float32)
    
    return pose_tensor

def csv_to_tensor(csv_file):
    """
    Convert CSV data to a tensor.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        drink_annotation_tensor: A Tensor of shape (N, T) where:
            N = number of participants
            T = number of frames
    """
    # Read the CSV file
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=0)

    pose_tensor = torch.tensor(data, dtype=torch.float32)  # shape: (T, N)
    return pose_tensor
