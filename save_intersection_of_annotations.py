import torch
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

NUM_KEYPOINTS = 17

def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["annotations"]["skeletons"]

def get_participant_ids_from_csv(csv_path):
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        return [pid.strip() for pid in first_line.split(',')]

def load_binary_matrix(csv_path):
    mat = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    # force 2D
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    return mat

# Prepare containers
# we'll collect across *all* files, then stack into big tensors
pose_samples = []      # list of torch.Tensor [17×2]
label_samples = []     # list of 0/1 floats
meta = []              # list of (video,segment,camera,participant,frame)

pose_dir   = "../annotations/pose/coco"
binary_dir = "../annotations/actions/drinking/No_Audio"

# Loop over each segment
for pose_path in Path(pose_dir).rglob("*.json"):
    # parse metadata from filename
    cam_s, vid_s, seg_s, _ = pose_path.stem.split("_")
    camera  = cam_s.replace("cam", "")
    video   = vid_s.replace("vid", "")
    segment = seg_s.replace("seg", "")

    # load pose frames & find matching annotation files
    pose_frames = load_pose_json(pose_path)
    matches = list(Path(binary_dir).rglob(f"vid{video}_seg{segment}_*.csv"))
    if not matches:
        continue

    for csv_path in matches:
        participant_ids = get_participant_ids_from_csv(csv_path)
        bin_mat = load_binary_matrix(csv_path)  # shape: (T, N)

        # for each frame & each pid, only append if both exist
        for t, frame in enumerate(pose_frames):
            for col, pid in enumerate(participant_ids):
                # 1) check annotation exists
                if t >= bin_mat.shape[0]:
                    continue
                label = bin_mat[t, col]
                # 2) check pose exists
                if pid not in frame:
                    continue

                # extract keypoints array
                keyps = frame[pid]["keypoints"]
                keyps = [float(x) if x is not None else float("nan") for x in keyps]
                keyps = np.array(keyps).reshape(NUM_KEYPOINTS, 2)

                # store
                pose_samples.append(torch.from_numpy(keyps).float())  # [17,2]
                label_samples.append(float(label))                    # 0.0 or 1.0
                meta.append((video, segment, camera, pid, t))

# Stack into big tensors
# X: [S, 17, 2],  Y: [S], where S = total number of matched samples
X = torch.stack(pose_samples)           # shape: (S, K, 2)
Y = torch.tensor(label_samples)         # shape: (S,)

print("Sampled", X.shape[0], "frame-participant pairs.")
print("X:", X.shape,   "  Y:", Y.shape)
