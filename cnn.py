import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

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

        assert K * D == 8 * 2, f"Expected /-17-/ 5 keypoints and 2 dims, got shape {x.shape}"
        if T < 8:
            raise ValueError(f"Input sequence too short: T={T}, but kernel size is 5")
        
        x = x.view(B, T, K * D)        # [B, T, 34]
        x = x.permute(0, 2, 1)         # [B, 34, T]

        x = F.relu(self.bn1(self.conv1(x)))  # [B, hidden, T]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, hidden, T]
        x = self.pool(x).squeeze(-1)         # [B, hidden]
        out = self.fc(x) #torch.sigmoid(self.fc(x))      # [B, 1]
        return out.squeeze(-1)               # [B]

# Training function for one epoch
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        X, y, meta = batch

        if torch.isnan(X).any():
            raise ValueError("NaN found in input features X! Please clean your dataset.")
        if torch.isnan(y).any(): # Though less likely for 0/1 labels
            raise ValueError("NaN found in input labels y! Please clean your dataset.")
        
        X = X.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        # Clip gradients (optional?)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:  # batch yields (X, y)
            X, y = batch
            X = X.to(device)
            y_true = y.to(device).float()

            outputs = model(X)  # Raw logits

            if not torch.isnan(outputs).any():
                loss = loss_fn(outputs, y_true)
                if not torch.isnan(loss):
                    total_loss += loss.item() * X.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_true_labels.append(y_true.cpu())
            all_predicted_labels.append(preds.cpu())

    avg_loss = float('nan')
    accuracy = float('nan')
    precision = float('nan')
    recall = float('nan')
    f1 = float('nan')

    if not all_true_labels:
        return avg_loss, accuracy, precision, recall, f1

    all_true_labels = torch.cat(all_true_labels).numpy()
    all_predicted_labels = torch.cat(all_predicted_labels).numpy()

    num_samples_for_loss = len(all_true_labels)

    if num_samples_for_loss > 0:
        avg_loss = total_loss / num_samples_for_loss

    if len(all_true_labels) > 0:
        accuracy = (all_predicted_labels == all_true_labels).sum() / len(all_true_labels)
        p, r, f, _ = precision_recall_fscore_support(
            all_true_labels, all_predicted_labels, average='binary', pos_label=1, zero_division=0
        )
        precision, recall, f1 = p, r, f

    return avg_loss, accuracy, precision, recall, f1

# Modified version of eval including mask 
# def evaluate(model, dataloader, loss_fn, device):
#     model.eval()
#     total_loss = 0.0

#     all_true_labels = []
#     all_predicted_labels = []

#     with torch.no_grad():
#         for batch in dataloader: # Assuming batch yields (X_coords, X_mask, Y_label)
#             X_coords_b, X_mask_b, y_b = batch
#             X_coords_b = X_coords_b.to(device)
#             X_mask_b = X_mask_b.to(device)
#             y_true = y_b.to(device).float()

#             outputs = model(X_coords_b, X_mask_b) # Raw logits

#             # Loss calculation (ensure valid outputs before calculating loss)
#             if not torch.isnan(outputs).any():
#                 loss = loss_fn(outputs, y_true)
#                 if not torch.isnan(loss):
#                     total_loss += loss.item() * X_coords_b.size(0)

#             probs = torch.sigmoid(outputs)
#             preds = (probs > 0.5).float()

#             all_true_labels.append(y_true.cpu())
#             all_predicted_labels.append(preds.cpu())

#     avg_loss = float('nan')
#     accuracy = float('nan')
#     precision = float('nan')
#     recall = float('nan')
#     f1 = float('nan')

#     if not all_true_labels: # No data processed
#         return avg_loss, accuracy, precision, recall, f1

#     all_true_labels = torch.cat(all_true_labels).numpy()
#     all_predicted_labels = torch.cat(all_predicted_labels).numpy()

#     num_samples_for_loss = len(all_true_labels) # Or however you define the denominator for loss

#     if num_samples_for_loss > 0 :
#         avg_loss = total_loss / num_samples_for_loss


#     if len(all_true_labels) > 0:
#         accuracy = (all_predicted_labels == all_true_labels).sum() / len(all_true_labels)
#         # Calculate precision, recall, F1 for the positive class (label 1)
#         # 'binary' assumes positive class is 1. Or use labels=[1], average=None and pick.
#         p, r, f, _ = precision_recall_fscore_support(
#             all_true_labels, all_predicted_labels, average='binary', pos_label=1, zero_division=0
#         )
#         precision, recall, f1 = p, r, f

#     return avg_loss, accuracy, precision, recall, f1

# Main training loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        # print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        # print(f"  Precision : {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
