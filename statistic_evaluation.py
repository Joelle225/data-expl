import torch
import numpy as np
import torch
import matplotlib.pyplot as plt

def normalize_position_Data(tensor):
    return tensor

def split_pose_by_annotation(pose_tensor, annotation_tensor):
    """
    Splits pose_tensor into two based on annotation_tensor (0 or 1 labels).
    
    Returns:
        pose_yes: Tensor of shape (M1, K, 2) where annotation == 1
        pose_no: Tensor of shape (M0, K, 2) where annotation == 0
    """
    N, T, K, _ = pose_tensor.shape

    # TODO: Remove rows where participant numbers not in both files

    # Flatten participant and frame dimensions to align both tensors
    pose_flat = pose_tensor.view(-1, K, 2)           # shape: (N*T, K, 2)
    annotation_flat = annotation_tensor.view(-1)     # shape: (N*T,)

    # Find mask indices
    yes_mask = annotation_flat == 1
    no_mask = annotation_flat == 0

    # Apply masks
    pose_yes = pose_flat[yes_mask]  # shape: (M1, K, 2)
    pose_no = pose_flat[no_mask]    # shape: (M0, K, 2)

    return pose_yes, pose_no


def mean_hand_head_distance(pose_tensor):
    """
    Computes mean Euclidean distances between head (1) and both hands (8, 9)
    for a pose tensor of shape (M, K, 2), where M = total relevant frames.
    
    Returns:
        mean_left_dist: Mean distance head↔left hand
        mean_right_dist: Mean distance head↔right hand
    """
    head = pose_tensor[:, 1]      # shape: (M, 2)
    left_hand = pose_tensor[:, 8]
    right_hand = pose_tensor[:, 9]

    # Compute Euclidean distances
    left_dist = torch.norm(head - left_hand, dim=1)    # shape: (M,)
    right_dist = torch.norm(head - right_hand, dim=1)  # shape: (M,)

    # Exclude NaNs if any
    left_dist = left_dist[~torch.isnan(left_dist)]
    right_dist = right_dist[~torch.isnan(right_dist)]

    return left_dist.mean().item(), right_dist.mean().item()

def compute_hand_head_distances(pose_tensor):
    """
    Compute per-frame distances between head (1) and hands (8, 9).
    
    Returns:
        left_dist (Tensor): distances head ↔ left wrist
        right_dist (Tensor): distances head ↔ right wrist
    """
    head = pose_tensor[:, 1]        # (M, 2)
    left_hand = pose_tensor[:, 8]
    right_hand = pose_tensor[:, 9]

    left_dist = torch.norm(head - left_hand, dim=1)
    right_dist = torch.norm(head - right_hand, dim=1)

    # Remove NaNs
    left_dist = left_dist[~torch.isnan(left_dist)]
    right_dist = right_dist[~torch.isnan(right_dist)]

    return left_dist, right_dist


def plot_hand_head_histograms(pose_yes, pose_no, bins=30):
    """
    Plot histograms of head-hand distances for YES and NO examples.
    """
    yes_left, yes_right = compute_hand_head_distances(pose_yes)
    no_left, no_right = compute_hand_head_distances(pose_no)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].hist(yes_left.numpy(), bins=bins, alpha=0.6, label="Left Hand (1-8)")
    axes[0].hist(yes_right.numpy(), bins=bins, alpha=0.6, label="Right Hand (1-9)")
    axes[0].set_title("Drinking")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(no_left.numpy(), bins=bins, alpha=0.6, label="Left Hand (1-8)")
    axes[1].hist(no_right.numpy(), bins=bins, alpha=0.6, label="Right Hand (1-9)")
    axes[1].set_title("Not Drinking")
    axes[1].set_xlabel("Distance")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
