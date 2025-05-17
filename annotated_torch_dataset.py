import torch
from torch.utils.data import Dataset

class SlidingWindowPoseDataset(Dataset):
    def __init__(self, sequences, window_size=45, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        for seq in sequences:
            X = seq['X']   # [T, 17, 2]
            Y = seq['Y']   # [T]
            meta = seq['meta']

            # Divide into window sized samples/chunks of sequences
            T = X.shape[0]
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                X_win = X[start:end]          # [window_size, 17, 2]
                Y_win = Y[start:end]          # [window_size]

                # Define window label: 1 if any frame is a 1
                window_label = torch.any(Y_win > 0).float()

                self.samples.append({
                    'X': X_win,               # [W, 17, 2]
                    'Y': window_label,        # 0.0 or 1.0
                    'meta': {
                        **meta,
                        'start_frame': meta['frames'][start],
                        'end_frame': meta['frames'][end - 1]
                    }
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['X'], sample['Y'] # Return the data and label to the loader
