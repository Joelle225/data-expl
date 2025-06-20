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
        seed=42,
        percentage_pos=0.6
    ):
        self.window_size = window_size
        self.stride = stride
        self.jitter_max = jitter_max
        self.reverse_positives = reverse_positives
        self.balance = balance
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.samples = []
        self.percentage_pos=percentage_pos

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
                label = (Y_win.float().mean() >= percentage_pos).float()

                if label == 0 and torch.any(Y_win > 0): #if overall label is 0, and there are positive labels in this window, discard, only use fully zero drinking windows for negatives during training
                    continue

                select_keypoints = [1, 2, 3, 5, 6, 8, 9, 12] # selects head shoulders and hands
                newX = X[start:end][:, select_keypoints, :]
                sample = {
                    'X': newX,      
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
        meta = sample['meta']

        return X.nan_to_num(nan=1.0), Y.nan_to_num(nan=1.0), meta 
