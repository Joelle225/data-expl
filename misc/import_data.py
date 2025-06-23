import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

NUM_KEYPOINTS = 17

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

def load_pose_file(json_path, video, segment, camera):
    """
    Load one JSON file of pose data and return a DataFrame.
    Format: (video, segment, camera, participant, frame, keypoint, [x, y])
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    skeletons = data["annotations"]["skeletons"]
    rows = []

    for frame_idx, frame in enumerate(skeletons):
        for pid, pose_data in frame.items():
            keypoints = pose_data["keypoints"]
            keypoints = [x if x is not None else float('nan') for x in keypoints]
            keypoints = np.array(keypoints).reshape(NUM_KEYPOINTS, 2)
            for kpt_idx in range(NUM_KEYPOINTS):
                xy = keypoints[kpt_idx]
                rows.append((video, segment, camera, pid, (frame_idx + 1), kpt_idx, xy))

    dframe = pd.DataFrame(rows, columns=["video", "segment", "camera", "participant", "frame", "keypoint", "xy"])
    print(dframe.head())
    return dframe

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

def load_binary_annotation(csv_path, video, segment, annotator):
    """
    Load one CSV annotation file and return a DataFrame.
    Assumes shape (T, N) with participants along columns.
    """

    # TODO: use header to index on participant
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        participant_ids = first_line.split(',')

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=0)
    
    if(data.ndim == 1):
        num_frames = len(data)
        num_participants = 1
    else:
        (num_frames, num_participants) = data.shape

    assert len(participant_ids) == num_participants, f"Mismatch between header and data shape in {csv_path}"

    rows = []
    for pid_idx in range(num_participants):
        pid = participant_ids[pid_idx].strip()
        for frame_idx in range(num_frames):
            flag = data[frame_idx, pid_idx] if num_participants > 1 else data[frame_idx]
            rows.append((video, segment, annotator, pid, frame_idx, flag))

    return pd.DataFrame(rows, columns=["video", "segment", "annotator", "participant", "frame", "drinking_flag"])

def batch_load_pose_jsons(pose_dir):
    all_pose_rows = []

    for file_path in Path(pose_dir).rglob("*.json"):
        # Extract metadata from filename or path
        video = file_path.stem.split("_")[1].replace("vid", "")
        segment = file_path.stem.split("_")[2].replace("seg", "")
        camera = file_path.stem.split("_")[0].replace("cam", "")

        pose_df = load_pose_file(file_path, video, segment, camera)
        all_pose_rows.append(pose_df)

    return pd.concat(all_pose_rows, ignore_index=True)

def batch_load_binary_csvs(ann_dir):
    all_ann_rows = []

    for file_path in Path(ann_dir).rglob("*.csv"):
        video = file_path.stem.split("_")[0].replace("vid", "")
        segment = file_path.stem.split("_")[1].replace("seg", "")
        annotator = file_path.stem.split("_")[2].replace("ann", "")

        ann_df = load_binary_annotation(file_path, video, segment, annotator)
        all_ann_rows.append(ann_df)

    return pd.concat(all_ann_rows, ignore_index=True)

pose_df = batch_load_pose_jsons("../annotations/pose/coco")
# ann_df  = batch_load_binary_csvs("../annotations/actions/drinking/No_Audio")

print(pose_df.head())
# → columns: [video, segment, camera, participant, frame, keypoint, x, y]

# print(ann_df.head())
# → columns: [video, segment, annotator, participant, frame, drinking_flag]
