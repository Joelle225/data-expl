# vid3_seg1_ann1.csv
# cam2_vid3_seg3_coco.json

# todo filter out the participants not in both files
# todo make sure input format to cnn will be compatible
# todo seperate by participant and pair with participant annotations

import import_data as im
import torch
import numpy as np
import matplotlib.pyplot as plt
import data_visualisation
import statistic_evaluation as se

pose_tensor = im.pose_json_to_tensor("../annotations/pose/coco/cam2_vid3_seg3_coco.json")
print(torch.min(pose_tensor), torch.max(pose_tensor))
# visualize_participant_to_images(pose_tensor, participant_index=0, output_dir="frames", max_size_mb=1000)

drink_annotation_tensor = im.csv_to_tensor("../annotations/actions/drinking/No_Audio/vid3_seg1_ann1.csv")
print(drink_annotation_tensor)

# MEAN DIFFS
# pose_yes, pose_no = split_pose_by_annotation(pose_tensor, annotation_tensor)
# yes_left, yes_right = mean_hand_head_distance(pose_yes)
# no_left, no_right = mean_hand_head_distance(pose_no)
# print(f"✅ YES:  Head–LeftHand = {yes_left:.4f}, Head–RightHand = {yes_right:.4f}")
# print(f"🚫 NO:   Head–LeftHand = {no_left:.4f}, Head–RightHand = {no_right:.4f}")

# pose_tensor = torch.randn(10, 100, 17, 2)  # Example pose tensor
# annotation_tensor = torch.randint(0, 2, (10, 100))  # Example annotation tensor

pose_yes, pose_no = se.split_pose_by_annotation(pose_tensor, drink_annotation_tensor)
se.plot_hand_head_histograms(pose_yes, pose_no)


# For each pose file, load it and get the tensor

# The pose data should be (video, segment, camera, participant, frame, keypoint, 2 -> for x & y)

# The drinking data should be (video, segment, annotator, participant, frames)
