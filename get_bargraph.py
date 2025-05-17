from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import defaultdict

NUM_KEYPOINTS = 17 

def distance(a, b):
    return np.linalg.norm(a - b)

def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["annotations"]["skeletons"]

def extract_keypoint_distance(pose_array, k1, k2):
    # Pose is shape (NUM_KEYPOINTS, 2) for one frame/person
    a, b = pose_array[k1], pose_array[k2]
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    return distance(np.array(a), np.array(b))

def get_participant_ids_from_csv(csv_path):
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        return [pid.strip() for pid in first_line.split(',')]

def load_binary_matrix(csv_path):
    return np.genfromtxt(csv_path, delimiter=',', skip_header=1)

# Accumulate results
distances = defaultdict(list)  # {'drinking': [...], 'not_drinking': [...]}

pose_dir = "../annotations/pose/coco"
binary_dir = "../annotations/actions/drinking/No_Audio"

for pose_path in Path(pose_dir).rglob("*.json"):
    filename = pose_path.stem
    parts = filename.split("_")  # e.g. ['cam2', 'vid2', 'seg8', 'coco']
    video = parts[1].replace("vid", "")
    segment = parts[2].replace("seg", "")
    camera = parts[0].replace("cam", "")
    
    # Match annotation file (e.g., vid2_seg8_ann1.csv)
    matching_csv = list(Path(binary_dir).rglob(f"vid{video}_seg{segment}_*.csv"))
    if not matching_csv:
        continue  # Skip if no matching annotation
    # csv_path = matching_csv[0]  # (choose first for now) TODO: multiple annotators

    for csv_path in matching_csv:
        pose_frames = load_pose_json(pose_path)
        participant_ids = get_participant_ids_from_csv(csv_path)
        binary_matrix = load_binary_matrix(csv_path)  # shape: (T, N)

        for frame_idx, frame in enumerate(pose_frames):
            for col_idx, pid in enumerate(participant_ids):
                drinking_flag = binary_matrix[frame_idx, col_idx] if binary_matrix.ndim > 1 else binary_matrix[frame_idx]
                if pid not in frame: # TODO: correctness?
                    continue # This participant not visible in this frame

                pose_data = frame[pid]["keypoints"]
                keypoints = [x if x is not None else float('nan') for x in pose_data]
                keypoints = np.array(keypoints).reshape(NUM_KEYPOINTS, 2)

                dist_lefth = extract_keypoint_distance(keypoints, 1, 9)
                dist_righth = extract_keypoint_distance(keypoints, 1, 8)

                label = "drinking" if drinking_flag == 1 else "not_drinking"
                if dist_lefth is not None and dist_lefth < 0.6 : distances[label].append(dist_lefth)
                if dist_righth is not None and dist_righth < 0.6: distances[label].append(dist_righth)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(distances["drinking"], bins=100, alpha=0.7, label="Drinking", color="blue")
plt.title("Distances hands to mouth (Drinking)")
plt.xlabel("Distance (normalized)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(distances["not_drinking"], bins=100, alpha=0.7, label="Not Drinking", color="green")
plt.title("Distances hands to mouth (Not Drinking)")
plt.xlabel("Distance (normalized)")

plt.tight_layout()
plt.show()
