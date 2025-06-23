import torch
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

# After running this file, dataset[] will look as follows:
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

NUM_KEYPOINTS = 17

# loader functions
def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["annotations"]["skeletons"]

def get_participant_ids_from_csv(csv_path):
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
    return [pid.strip() for pid in header]

def load_binary_matrix(csv_path):
    mat = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    return mat

# collect per-sequence data
# Mapping (video, segment, camera, participant, annotator) -> list of (time, keyps, label)
seqs = defaultdict(list)

pose_dir   = "../annotations/pose/coco"
binary_dir = "../annotations/actions/drinking/No_Audio"

for pose_path in Path(pose_dir).rglob("*.json"):
    # parse pose metadata
    cam_s, vid_s, seg_s, _ = pose_path.stem.split("_")
    camera  = cam_s.replace("cam", "")
    video   = vid_s.replace("vid", "")
    segment = seg_s.replace("seg", "")

    pose_frames = load_pose_json(pose_path)
    # find all annotator CSVs for this video/segment
    matches = list(Path(binary_dir).rglob(f"vid{video}_seg{segment}_*.csv"))
    if not matches:
        continue

    for csv_path in matches:
        # extract annotator from filename (e.g., vid2_seg8_ann1.csv)
        annotator = Path(csv_path).stem.split('_')[-1].replace("ann", "")
        participant_ids = get_participant_ids_from_csv(csv_path)
        bin_mat = load_binary_matrix(csv_path)

        for frame_num, frame in enumerate(pose_frames):
            if frame_num >= bin_mat.shape[0]:
                break
            for id_num, participant_id in enumerate(participant_ids):
                if participant_id not in frame:
                    continue
                label = float(bin_mat[frame_num, id_num])
                keypts = frame[participant_id]["keypoints"]
                keypts = [float(x) if x is not None else float("nan") for x in keypts]
                keypts = np.array(keypts).reshape(NUM_KEYPOINTS, 2)
                seqs[(video, segment, camera, participant_id, annotator)].append((frame_num, keypts, label))

# build sequence-level tensors
# container: list of dicts with 'X', 'Y', 'meta'
dataset = []
for (video, segment, camera, pid, annotator), entries in seqs.items():
    entries.sort(key=lambda e: e[0])
    times, keyps_list, labels = zip(*entries)
    Xseq = torch.tensor(np.stack(keyps_list), dtype=torch.float32) # [T,17,2]
    Yseq = torch.tensor(labels, dtype=torch.float32) # [T]
    meta = {
        'video': video,
        'segment': segment,
        'camera': camera,
        'participant': pid,
        'annotator': annotator,
        'frames': list(times)
    }
    dataset.append({'X': Xseq, 'Y': Yseq, 'meta': meta})

print(f"Built dataset with {len(dataset)} sequences of {len(dataset[0]['X'])}.")

# save dataset to disk 
# Use torch.save to persist the entire dataset list
output_path = Path("./drinking_sequence_dataset.pth")
torch.save(dataset, output_path)
print(f"Dataset saved to {output_path.resolve()}")

# loading snippet
# Later, in another script, load with:
# 
#   import torch
#   from pathlib import Path
#   dataset = torch.load(Path("/path/to/drinking_sequence_dataset.pth"))
#   # dataset is a list of dicts with keys 'X', 'Y', 'meta'
#
# Then wrap dataset into a PyTorch Dataset/ DataLoader as needed.

print("saved to disk... done!")
