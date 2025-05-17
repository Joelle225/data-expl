# import matplotlib.pyplot as plt
# import matplotlib
# import json
# import import_data as im
# import numpy as np
# import torch
# matplotlib.use("Qt5Agg")  # Or "Qt5Agg" depending on your system

# CUSTOM_SKELETON = [
#     (0, 1),  # head → nose
#     (1, 2),  # nose → neck
#     (2, 3), (3, 4), (4, 5),  # neck → RShoulder → RElbow → RWrist
#     (2, 6), (6, 7), (7, 8),  # neck → LShoulder → LElbow → LWrist
#     (2, 9), (9, 10), (10, 11),  # neck → RHip → RKnee → RAnkle
#     (2, 12), (12, 13), (13, 14),  # neck → LHip → LKnee → LAnkle
#     (11, 15),  # RAnkle → RFoot
#     (14, 16),  # LAnkle → LFoot
# ]

# def visualize_participant_over_time(pose_tensor, participant_index=0, interval=100):
#     num_frames = pose_tensor.shape[1]
#     num_keypoints = pose_tensor.shape[2]

#     for t in range(num_frames):
#         keypoints = pose_tensor[participant_index, t]  # Shape: (K, 2)

#         if torch.isnan(keypoints).all():
#             continue  # Skip if all keypoints are missing in this frame

#         x = keypoints[:, 0].numpy()
#         y = keypoints[:, 1].numpy()

#         plt.figure(figsize=(5, 5))
#         plt.scatter(x, y, c="blue")
        
#         for idx, (x_i, y_i) in enumerate(zip(x, y)):
#             if not np.isnan(x_i) and not np.isnan(y_i):
#                 plt.text(x_i, y_i, str(idx), fontsize=8, color="red")

#         # Draw skeleton lines
#         for (i, j) in CUSTOM_SKELETON:
#             if i < num_keypoints and j < num_keypoints:
#                 if not np.isnan(x[i]) and not np.isnan(x[j]) and not np.isnan(y[i]) and not np.isnan(y[j]):
#                     plt.plot([x[i], x[j]], [y[i], y[j]], 'k-')

#         plt.gca().invert_yaxis()
#         plt.title(f"Participant {participant_index}, Frame {t}")
#         plt.axis("equal")
#         plt.pause(interval / 1000.0)  # Pause to simulate animation
#         plt.clf()

#     plt.ion()  # Turn on interactive mode
#     plt.show()
    
#     plt.close()

# def run_vis():
#     pose_tensor = im.pose_json_to_tensor("../annotations/pose/coco/cam8_vid2_seg9_coco.json")

#     # Call this function to run the visualization
#     visualize_participant_over_time(pose_tensor, participant_index=0, interval=200)

# run_vis()

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import import_data as im

CUSTOM_SKELETON = [
    (1, 5), # (head -> Neck)
    (5, 4), (5, 3), # Neck -> Shoulders
    (4, 7), (7, 9), # arms left
    (3, 6), (6, 8), # arms right 
    (5, 11), (5, 10), # neck -> hips
    (11, 13), (13, 15), # legs left
    (10, 12), (12, 14) # legs right
]

def get_folder_size(folder):
    """Returns folder size in bytes."""
    return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())

def visualize_participant_to_images(pose_tensor, participant_index=0, output_dir="frames", max_size_mb=100):
    os.makedirs(output_dir, exist_ok=True)
    max_size_bytes = max_size_mb * 1024 * 1024

    num_frames = pose_tensor.shape[1]
    num_keypoints = pose_tensor.shape[2]

    for t in range(num_frames):
        keypoints = pose_tensor[participant_index, t]

        if torch.isnan(keypoints).all():
            continue

        x = keypoints[:, 0].numpy() # is this wrong?
        y = keypoints[:, 1].numpy()

        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, c="blue")

        for idx, (x_i, y_i) in enumerate(zip(x, y)):
            if not np.isnan(x_i) and not np.isnan(y_i):
                plt.text(x_i, y_i, str(idx + 1), fontsize=8, color="black")

        for (i, j) in CUSTOM_SKELETON:
            if i < num_keypoints and j < num_keypoints:
                if not np.isnan(x[i]) and not np.isnan(x[j]) and not np.isnan(y[i]) and not np.isnan(y[j]):
                    plt.plot([x[i-1], x[j-1]], [y[i-1], y[j-1]], 'k-')

        plt.gca().invert_yaxis()
        plt.title(f"Participant {participant_index}, Frame {t}")
        plt.axis("equal")

        frame_path = os.path.join(output_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path)
        plt.close()

        # Check folder size
        if get_folder_size(output_dir) > max_size_bytes:
            print(f"!! Folder size exceeded {max_size_mb}MB at frame {t}. Halting save.")
            break
