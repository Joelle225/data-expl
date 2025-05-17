import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from annotated_torch_dataset import SlidingWindowPoseDataset

# Load saved sequence dataset
sequence_dataset = torch.load("./drinking_sequence_dataset.pth")

# Create dataset and loader
windowed_dataset = SlidingWindowPoseDataset(sequence_dataset, window_size=45, stride=5)
loader = DataLoader(windowed_dataset, batch_size=4, shuffle=True)

# Get one batch
X_batch, Y_batch = next(iter(loader))  # X: [B, 45, 17, 2], Y: [B]

# Visualize the first few samples in the batch
for i in range(min(4, len(X_batch))):
    sequence = X_batch[i].numpy()     # shape: [45, 17, 2]
    label = int(Y_batch[i].item())    # 0 or 1

    fig, axes = plt.subplots(3, 15, figsize=(18, 4))
    axes = axes.flatten()

    for t in range(45):
        ax = axes[t]
        pose = sequence[t]  # [17, 2]
        x, y = pose[:, 0], -pose[:, 1]  # Flip Y-axis for display

        ax.scatter(x, y, c='r' if label else 'b')
        ax.plot(x, y, 'k-', alpha=0.2)
        ax.set_title(f"t={t}")
        ax.axis("off")

    fig.suptitle(f"Sample {i} - {'Drinking' if label else 'Not Drinking'}", fontsize=16)
    plt.tight_layout()
    plt.show()
