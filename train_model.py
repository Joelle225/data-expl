from sklearn.model_selection import KFold
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from annotated_torch_dataset import SlidingWindowPoseDataset
from cnn import DrinkingCNN, train_model
import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Load dataset
sequence_dataset = torch.load("./drinking_sequence_dataset.pth")

# Group sequences by participant-video-segment (ignoring camera to prevent leakage)
grouped = defaultdict(list)
for idx, seq in enumerate(sequence_dataset):
    meta = seq['meta']
    key = (meta['participant'], meta['video'], meta['segment'])
    grouped[key].append(idx)

# Create list of fold units (each unit is all cams of one segment)
group_keys = list(grouped.keys())
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store out-of-fold predictions and labels
all_val_preds = []
all_val_labels = []

for fold, (train_idx, val_idx) in enumerate(kf.split(group_keys)):
    print(f"\n=== Fold {fold + 1}/{n_splits} ===")

    # Get flattened list of indices for training/validation
    train_indices = [idx for i in train_idx for idx in grouped[group_keys[i]]]
    val_indices = [idx for i in val_idx for idx in grouped[group_keys[i]]]

    train_sequences = [sequence_dataset[i] for i in train_indices]
    val_sequences = [sequence_dataset[i] for i in val_indices]

    # Construct datasets -- TODO: Make sure no leakage due to duplicates between camera feeds 
    train_dataset = SlidingWindowPoseDataset(
        sequences=train_sequences,
        window_size=45,
        stride=1,
        neg_to_pos_ratio=4,
        balance=True,
        jitter_max=3,
        reverse_positives=True
    )

    val_dataset = SlidingWindowPoseDataset(
        sequences=val_sequences,
        window_size=45,
        stride=1,
        neg_to_pos_ratio=4,  # or False for full negatives
        balance=False,       # Evaluate on unbalanced validation
        jitter_max=0,
        reverse_positives=False
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model, optimizer, loss
    model = DrinkingCNN().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if len(train_dataset) > 0:
        train_labels_for_weight = [sample_tuple[2].item() for sample_tuple in train_dataset]
        num_pos_train = sum(1 for label in train_labels_for_weight if label == 1.0)
        num_neg_train = len(train_labels_for_weight) - num_pos_train

        if num_pos_train > 0:
            effective_pos_weight = torch.tensor(num_neg_train / num_pos_train, device=device)
            print(f"Using pos_weight for BCEWithLogitsLoss: {effective_pos_weight.item():.2f}")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=effective_pos_weight)
        else:
            print("Warning: No positive samples in training data for this fold. Using default BCEWithLogitsLoss.")
            loss_fn = torch.nn.BCEWithLogitsLoss() # Fallback
    else:
        print("Warning: Training dataset is empty for this fold.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=20
    )

    # Save weights per fold
    torch.save(model.state_dict(), f"cnn_model_fold{fold+1}.pth")

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch) #.squeeze(1)       # [B]
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y_batch.cpu().numpy()

            all_val_preds.append(probs)
            all_val_labels.append(labels)

# Concatenate all fold predictions
y_true = np.concatenate(all_val_labels)
y_scores = np.concatenate(all_val_preds)

# Evaluate metrics
roc_auc = roc_auc_score(y_true, y_scores)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_scores > 0.5, average="binary"
)

print("\n=== Ensemble Performance Across All Folds ===")
print(f"ROC AUC:     {roc_auc:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1 Score:    {f1:.4f}")
