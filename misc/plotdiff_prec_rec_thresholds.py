import os
from sklearn.model_selection import KFold
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score, precision_recall_curve
import datetime
import matplotlib.pyplot as plt
from annotated_torch_dataset import SlidingWindowPoseDataset
from cnn import DrinkingCNN, train_model

# Store metrics from each fold
all_roc_aucs = []
all_precisions = []
all_recalls = []
all_f1s = []

# New lists to store data for the new plot
all_fold_precisions_vs_thresh = []
all_fold_recalls_vs_thresh = []


######ooo######
# Knobs to turn:
n_splits                = 5
train_window_size       = 180
train_stride            = 60 # was 3
train_neg_to_pos_ratio  = 2 # was 10
train_balance_dataset   = True
train_jitter_max        = 0
train_reverse_positives = False # try setting to false to see what happens to performance TODO.
learning_rate           = 3e-4 # was 1e-3

val_window_size         = train_window_size # for now keep the same
val_stride              = 5
val_neg_to_pos_ratio    = 10                 # irrellevant
val_balance_dataset     = True

batch_size              = 32
bce_pos_weight_factor   = 4 # was 350
num_epochs              = 2 # Reduced for quick demonstration

######ooo######

# Misc Options
save_model_weights=True
plot_y_values = False
plot_2_curves = True
plot_pr_vs_threshold = True # <-- NEW: Flag to control the new plot generation

# Custom collate function (returning meta in dataloader won't work without this func) #

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of (data, label, meta).
    'meta' is a list of dicts and is returned as is.
    """
    X_list = [item[0] for item in batch]
    Y_list = [item[1] for item in batch]
    meta_list = [item[2] for item in batch]

    X_batch = torch.stack(X_list)
    Y_batch = torch.stack(Y_list)

    return X_batch, Y_batch, meta_list

# Load dataset - Creating a dummy dataset for demonstration
# In your actual code, you would use:
# sequence_dataset = torch.load("./drinking_sequence_dataset.pth")
num_sequences = 50
sequence_len = 1000
num_features = 10 # Example: 5 keypoints with (x, y) coords

sequence_dataset = torch.load("./drinking_sequence_dataset.pth")



# Group sequences by participant-video-segment (ignoring camera and annotator to prevent leakage)
grouped = defaultdict(list)
for idx, seq in enumerate(sequence_dataset):
    meta = seq['meta']
    key = (meta['participant'], meta['video'], meta['segment'])
    grouped[key].append(idx)

# Create list of fold units (each unit is all cams of one segment)
group_keys = list(grouped.keys())
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store out-of-fold predictions and labels for overall ensemble evaluation
overall_y_scores_accumulated = [] # Probabilities for positive class
overall_y_labels_accumulated = []


for fold, (train_idx, val_idx) in enumerate(kf.split(group_keys)):
    print(f"\n=== Fold {fold + 1}/{n_splits} ===")

    # Get flattened list of indices for training/validation
    train_indices = [idx for i in train_idx for idx in grouped[group_keys[i]]]
    val_indices = [idx for i in val_idx for idx in grouped[group_keys[i]]]

    train_sequences = [sequence_dataset[i] for i in train_indices]
    val_sequences = [sequence_dataset[i] for i in val_indices]

    # Construct datasets
    print("Loading training set")
    train_dataset = SlidingWindowPoseDataset(
        sequences=train_sequences,
        window_size=train_window_size,
        stride=train_stride,
        neg_to_pos_ratio=train_neg_to_pos_ratio,
        balance=train_balance_dataset,
        jitter_max=train_jitter_max,
        reverse_positives=train_reverse_positives
    )
    print(f"Training dataset size: {len(train_dataset)}")

    print("Loading validation set")
    val_dataset = SlidingWindowPoseDataset(
        sequences=val_sequences,
        window_size=val_window_size,
        stride=val_stride,
        neg_to_pos_ratio=val_neg_to_pos_ratio,
        balance=val_balance_dataset,
        jitter_max=0,
        reverse_positives=False,
        percentage_pos=0.3
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    # Model, optimizer, loss
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Skipping fold due to empty dataset.")
        continue

    sample_X, _, _ = train_dataset[0]
    # Correctly calculate input_channels: it's the number of features per time step.
    input_channel_size = sample_X.shape[0] # Features are channels in Conv1D
    model = DrinkingCNN(input_channels=input_channel_size).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_labels_for_weight = [sample_tuple[1].item() for sample_tuple in train_dataset]
    num_pos_train = sum(train_labels_for_weight)
    num_neg_train = len(train_labels_for_weight) - num_pos_train

    val_labels_for_weight = [sample_tuple[1].item() for sample_tuple in val_dataset]
    num_pos_val = sum(val_labels_for_weight)
    num_neg_val = len(val_labels_for_weight) - num_pos_val

    print(f"Num pos train this fold: {num_pos_train}, Num neg train this fold: {num_neg_train}")
    print(f"Num pos val this fold: {num_pos_val}, Num neg val this fold: {num_neg_val}")

    if num_pos_train > 0:
        effective_pos_weight = torch.tensor(bce_pos_weight_factor * (num_neg_train / num_pos_train), device=device)
        print(f"Using pos_weight for BCEWithLogitsLoss: {effective_pos_weight.item():.2f}")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=effective_pos_weight)
    else:
        print("Warning: No positive samples in training data. Using default BCEWithLogitsLoss.")
        loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs
    )

    # Save weights per fold
    if save_model_weights:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), f"cnn_model_fold{fold+1}_{timestamp}.pth")

    # --- Find the optimal threshold on the TRAINING data ---
    model.eval()
    train_probs_for_threshold = []
    train_labels_for_threshold = []
    with torch.no_grad():
        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            train_probs_for_threshold.append(probs)
            train_labels_for_threshold.append(y_batch.cpu().numpy())

    train_probs_for_threshold = np.concatenate(train_probs_for_threshold)
    train_labels_for_threshold = np.concatenate(train_labels_for_threshold)

    thresholds_f1 = np.arange(0.0, 1.0, 0.01)
    f1_scores = [f1_score(train_labels_for_threshold, (train_probs_for_threshold >= t).astype(int), zero_division=0) for t in thresholds_f1]
    best_threshold = thresholds_f1[np.argmax(f1_scores)]
    print(f"Best threshold for this fold (from training data): {best_threshold:.2f}")


    # --- Evaluate on the VALIDATION data ---
    model.eval()
    y_scores_this_fold = []
    y_true_this_fold = []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_scores_this_fold.append(probs)
            y_true_this_fold.append(y_batch.cpu().numpy())

    if not y_scores_this_fold:
        print("Validation set was empty, skipping metrics for this fold.")
        continue

    y_scores_this_fold = np.concatenate(y_scores_this_fold)
    y_true_this_fold = np.concatenate(y_true_this_fold)

    # Check if there are positive samples in the validation set for this fold
    if len(np.unique(y_true_this_fold)) < 2:
        print("Warning: Validation set has only one class. ROC AUC and other metrics are not well-defined.")
        # Assign default values or skip
        roc_auc_fold = np.nan
        precision_fold, recall_fold, f1_fold = 0.0, 0.0, 0.0
    else:
        y_pred_this_fold = (y_scores_this_fold >= best_threshold).astype(int)
        roc_auc_fold = roc_auc_score(y_true_this_fold, y_scores_this_fold)
        precision_fold, recall_fold, f1_fold, _ = precision_recall_fscore_support(
            y_true_this_fold, y_pred_this_fold, average="binary", zero_division=0
        )

    if not np.isnan(roc_auc_fold): all_roc_aucs.append(roc_auc_fold)
    else: print("Warn: NaN roc_auc detected!")
    all_precisions.append(precision_fold)
    all_recalls.append(recall_fold)
    all_f1s.append(f1_fold)

    print(f"\n--- Performance for Fold {fold + 1} ---")
    print(f"ROC AUC:     {roc_auc_fold:.4f}")
    print(f"Precision:   {precision_fold:.4f}")
    print(f"Recall:      {recall_fold:.4f}")
    print(f"F1 Score:    {f1_fold:.4f}")

    # Append results for overall ensemble calculation
    overall_y_scores_accumulated.append(y_scores_this_fold)
    overall_y_labels_accumulated.append(y_true_this_fold)

    # --- NEW: Generate Precision-Recall vs. Threshold plot for this fold ---
    if plot_pr_vs_threshold and len(np.unique(y_true_this_fold)) > 1:
        precisions, recalls, thresholds = precision_recall_curve(y_true_this_fold, y_scores_this_fold)
        all_fold_precisions_vs_thresh.append(np.interp(np.linspace(0, 1, 100), thresholds, precisions[:-1]))
        all_fold_recalls_vs_thresh.append(np.interp(np.linspace(0, 1, 100), thresholds, recalls[:-1]))

        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
        plt.plot(thresholds, recalls[:-1], label='Recall', color='green')
        plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best F1 Threshold ({best_threshold:.2f})')
        plt.title(f'Precision and Recall vs. Threshold - Fold {fold + 1}')
        plt.xlabel('Classification Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.ylim([-0.05, 1.05])
        plt.savefig(f"prec_recall_vs_threshold_fold_{fold+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()


# --- Overall Performance Across All Folds ---

print("\n--- Overall Cross-Validation Performance ---")
print(f"Average ROC AUC: {np.mean(all_roc_aucs):.4f} (+/- {np.std(all_roc_aucs):.4f})")
print(f"Average Precision: {np.mean(all_precisions):.4f} (+/- {np.std(all_precisions):.4f})")
print(f"Average Recall:    {np.mean(all_recalls):.4f} (+/- {np.std(all_recalls):.4f})")
print(f"Average F1 Score:  {np.mean(all_f1s):.4f} (+/- {np.std(all_f1s):.4f})")

# --- NEW: Plot average Precision and Recall vs. Threshold ---
if plot_pr_vs_threshold and all_fold_precisions_vs_thresh:
    mean_thresholds = np.linspace(0, 1, 100)

    mean_precisions = np.mean(all_fold_precisions_vs_thresh, axis=0)
    std_precisions = np.std(all_fold_precisions_vs_thresh, axis=0)
    precisions_upper = np.minimum(mean_precisions + std_precisions, 1)
    precisions_lower = np.maximum(mean_precisions - std_precisions, 0)

    mean_recalls = np.mean(all_fold_recalls_vs_thresh, axis=0)
    std_recalls = np.std(all_fold_recalls_vs_thresh, axis=0)
    recalls_upper = np.minimum(mean_recalls + std_recalls, 1)
    recalls_lower = np.maximum(mean_recalls - std_recalls, 0)

    plt.figure(figsize=(12, 8))

    # Plot Mean Precision
    plt.plot(mean_thresholds, mean_precisions, color='blue', label='Mean Precision')
    plt.fill_between(mean_thresholds, precisions_lower, precisions_upper, color='blue', alpha=0.2)

    # Plot Mean Recall
    plt.plot(mean_thresholds, mean_recalls, color='green', label='Mean Recall')
    plt.fill_between(mean_thresholds, recalls_lower, recalls_upper, color='green', alpha=0.2)

    plt.title('Mean Precision and Recall vs. Classification Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.savefig(f"mean_prec_recall_vs_threshold_curves_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
