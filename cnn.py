import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# A very simplified model, lared-laughted model could be adapted instead, perhaps.
class DrinkingCNN(nn.Module):
    def __init__(self, input_channels=34, hidden_channels=64, window_size=45):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_channels)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # x: [B, T, 17, 2] → flatten keypoints
        B, T, K, D = x.shape
        x = x.view(B, T, K * D)        # [B, T, 34]
        x = x.permute(0, 2, 1)         # [B, 34, T]

        x = F.relu(self.bn1(self.conv1(x)))  # [B, hidden, T]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, hidden, T]
        x = self.pool(x).squeeze(-1)         # [B, hidden]
        out = torch.sigmoid(self.fc(x))      # [B, 1]
        return out.squeeze(-1)               # [B]

# Training function for one epoch
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        X, y = batch
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)

# Evaluation function
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * X.size(0)

            # Binary accuracy
            preds = (outputs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# Main training loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
