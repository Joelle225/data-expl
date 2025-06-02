import torch
import numpy as np
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class SlidingWindowPoseDataset(Dataset):
    def __init__(
        self,
        sequences,
        window_size=45,
        stride=1,
        neg_to_pos_ratio=3,
        balance=True,
        jitter_max=5,
        reverse_positives=True,
        seed=42
    ):
        self.window_size = window_size
        self.stride = stride
        self.jitter_max = jitter_max
        self.reverse_positives = reverse_positives
        self.balance = balance
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.samples = []

        random.seed(seed)
        np.random.seed(seed)

        pos_samples = []
        neg_samples = []

        for seq in tqdm(sequences, desc="Initializing Dataset"):
            X = seq['X']   # [T, 17, 2]
            Y = seq['Y']   # [T]
            meta = seq['meta']
            T = X.shape[0]

            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                Y_win = Y[start:end]
                label = torch.any(Y_win > 0).float()

                select_keypoints = [0, 2, 3, 7, 8] # selects head shoulders and hands
                newX = X[start:end][:, select_keypoints, :]
                sample = {
                    'X': newX,       # !!!HI, HERE THIS COMMENT IS IMPORTANT!!! I want to turn this [W, 17, 2] --> into extraacted features instead. How?
                    'Y': label,              # float (0.0 or 1.0)
                    'meta': {
                        **meta,
                        'start_frame': meta['frames'][start],
                        'end_frame': meta['frames'][end - 1],
                        'start_idx': start
                    }
                }

                if label == 1.0:
                    pos_samples.append(sample)
                    if reverse_positives:
                        # Add reversed version of window
                        rev_sample = {
                            'X': torch.flip(X[start:end][:, select_keypoints ,:], dims=[0]),
                            'Y': label,
                            'meta': {**sample['meta'], 'reversed': True}
                        }
                        pos_samples.append(rev_sample)
                else:
                    neg_samples.append(sample)

        if balance:
            keep_n_neg = min(len(neg_samples), len(pos_samples) * neg_to_pos_ratio)
            random.shuffle(neg_samples)
            self.samples = pos_samples + neg_samples[:keep_n_neg]
        else:
            self.samples = pos_samples + neg_samples

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = sample['X']
        Y = sample['Y']
        # meta = sample['meta']

        return X.nan_to_num(nan=1.0), Y.nan_to_num(nan=1.0)
