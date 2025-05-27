from torch.utils.data import DataLoader
from annotated_torch_dataset import SlidingWindowPoseDataset
import torch
from torch import nn
from cnn import DrinkingCNN, train_model

# Load your saved dataset from disk
sequence_dataset = torch.load("./drinking_sequence_dataset.pth")

# Wrap with your custom Dataset
# windowed_dataset = SlidingWindowPoseDataset(sequence_dataset, window_size=45, stride=5)

dataset = SlidingWindowPoseDataset(
    sequences=sequence_dataset,   # ← raw [T, 17, 2] pose data
    window_size=45,
    stride=1,
    neg_to_pos_ratio=4,
    balance=True,
    jitter_max=3,
    reverse_positives=True
)

# Wrap with DataLoader
train_loader, val_loader = DataLoader(
    dataset,
    batch_size=32,          # Number of samples per batch
    shuffle=True,           # Shuffle for training randomness
    num_workers=4,          # Parallel data loading (can use 0 for debugging)
    pin_memory=True)        # Speed optimization when using CUDA



# Example of using it
# for X_batch, y_batch in train_loader:       # Sizes:
#     print("Batch X shape:", X_batch.shape)  # [32, 45, 17, 2]
#     print("Batch Y shape:", y_batch.shape)  # [32]
#     break  # Only show one batch for now

# Assume:
model = DrinkingCNN()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Go train!
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    num_epochs=20)

# Save weights for later comparison / progress plotting
torch.save(model.state_dict(), "cnn_model.pth")
