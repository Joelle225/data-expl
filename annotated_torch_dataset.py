import torch
import numpy as np
from torch.utils.data import Dataset
import random

class SlidingWindowPoseDataset(Dataset):
    def __init__(
        self,
        sequences,
        window_size=45,
        stride=1,
        neg_to_pos_ratio=4,
        balance=True,
        reverse_positives=True,
        seed=42,
        is_for_sklearn=False,
        min_1_label=0.30
    ):
        self.window_size = window_size
        self.stride = stride
        self.reverse_positives = reverse_positives
        self.balance = balance
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.samples = []
        self.is_for_sklearn = is_for_sklearn
        self.min_1_label=min_1_label

        random.seed(seed)
        np.random.seed(seed)

        pos_samples = [sample for sample in sequences if sample['Y'] == 1]
        pos_samples.extend(pos_samples)
        neg_samples = [sample for sample in sequences if sample['Y'] == 0]

        if balance:
            target_n_pos = len(pos_samples)
            if target_n_pos == 0 and len(neg_samples) > 0: # No positive samples
                 print(f"Warning: No positive samples found. Using all {len(neg_samples)} negative samples (unbalanced).")
                 self.samples = neg_samples # Or decide to keep a subset
            elif target_n_pos == 0 and len(neg_samples) == 0: # No samples at all
                 print("Warning: No positive or negative samples found.")
                 self.samples = []
            else: # Positive and Negative samples exist
                keep_n_neg = min(len(neg_samples), target_n_pos * neg_to_pos_ratio)
                random.shuffle(neg_samples)
                self.samples = pos_samples + neg_samples[:keep_n_neg]
        else:
            self.samples = pos_samples + neg_samples

        if self.samples: # Only shuffle if there are samples
            random.shuffle(self.samples)
        else:
            print("Warning: Dataset is empty after processing and balancing.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        X_data = sample['X']
        Y_data = sample['Y']

        # Features are already NumPy arrays, replace NaNs if any
        X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0) # RF can't handle NaNs
        return X_data, Y_data # Y_data is already a float 0.0 or 1.0

