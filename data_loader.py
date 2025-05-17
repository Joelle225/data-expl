from torch.utils.data import DataLoader
from annotated_torch_dataset import SlidingWindowPoseDataset
import torch

# Load your saved dataset from disk
sequence_dataset = torch.load("./drinking_sequence_dataset.pth")

# Wrap with your custom Dataset
windowed_dataset = SlidingWindowPoseDataset(sequence_dataset, window_size=45, stride=5)

# Wrap with DataLoader
train_loader = DataLoader(
    windowed_dataset,
    batch_size=32,          # Number of samples per batch
    shuffle=True,           # Shuffle for training randomness
    num_workers=4,          # Parallel data loading (can use 0 for debugging)
    pin_memory=True)        # Speed optimization when using CUDA

# Example of using it
for X_batch, y_batch in train_loader:       # Sizes:
    print("Batch X shape:", X_batch.shape)  # [32, 45, 17, 2]
    print("Batch Y shape:", y_batch.shape)  # [32]
    break  # Only show one batch for now

