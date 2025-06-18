from sklearn.model_selection import KFold
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from annotated_torch_dataset import SlidingWindowPoseDataset
from cnn import DrinkingCNN, train_model
import torch
# from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score
import datetime

############
### TODO's
## 1. array size too big
## 2. step size to erratic/big
## 3. feature extraction
## *4. TQDM in data init
## 5. refactor into neater code
## 6. fix leak in train/val split on annotator as well
## 7. with this little data, should I even use a CNN?
## 8. add noise to the positives and re-add them into the sampler
# TODO try gigantic window size

## Attention: TODO check if use correct: PyTorch's Conv1d typically expects (batch_size, channels, sequence_length), so (batch_size, N, W) if N is your number of feature channels
############

######ooo######
# Knobs to turn: 
n_splits                = 5
train_window_size       = 200
train_stride            = 20 # was 3
train_neg_to_pos_ratio  = 2 # was 10
train_balance_dataset   = True 
train_jitter_max        = 0
train_reverse_positives = False # try setting to false to see what happens to performance TODO.
learning_rate           = 3e-4 # was 1e-3

val_window_size         = train_window_size # for now keep the same
val_stride              = 20
val_neg_to_pos_ratio    = 4                 # irrellevant
val_balance_dataset     = False

batch_size              = 32
bce_pos_weight_factor   = 10 # was 350
num_epochs              = 15
######ooo######

# Misc Options
save_model_weights=True

# Load dataset
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

# group_sizes = [len(grouped[k]) for k in group_keys]
# print("Group sizes (number of samples per group):", group_sizes)
# print("Total samples:", sum(group_sizes))
# exit()

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

    # Construct datasets -- TODO: Make sure no leakage due to duplicates between camera feeds 
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
        neg_to_pos_ratio=val_neg_to_pos_ratio,  # or False for full negatives
        balance=val_balance_dataset,            # Evaluate on unbalanced validation
        jitter_max=0,
        reverse_positives=False
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, optimizer, loss
    sample_X, _ = train_dataset[0]
    input_channel_size = sample_X.shape[1] * sample_X.shape[2]
    model = DrinkingCNN(input_channels=input_channel_size).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if len(train_dataset) > 0:
        train_labels_for_weight = [sample_tuple[1].item() for sample_tuple in train_dataset]
        num_pos_train = sum(1 for label in train_labels_for_weight if label == 1.0)
        num_neg_train = len(train_labels_for_weight) - num_pos_train
        print(f"Num pos this fold: {num_pos_train}, Num negative this fold: {num_neg_train}")

        if num_pos_train > 0:
            effective_pos_weight = torch.tensor(bce_pos_weight_factor * (num_neg_train / num_pos_train), device=device) # device=device) #
            print(f"Using pos_weight for BCEWithLogitsLoss: {effective_pos_weight.item():.2f}")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=effective_pos_weight)
        else:
            print("Warning: No positive samples in training data for this fold. Using default BCEWithLogitsLoss.")
            loss_fn = torch.nn.BCEWithLogitsLoss() # Fallback
    else:
        print("Warning: Training dataset is empty for this fold.")

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
    # if save_model_weights: 
    #     torch.save(model.state_dict(), f"cnn_model_fold{fold+1}.pth")
    # if save_model_weights:
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     torch.save(model.state_dict(), f"cnn_model_fold{fold+1}_{timestamp}.pth")

    # Eval model for this fold
    model.eval()
    fold_batch_probs = [] # Store probabilities from each batch in this fold
    fold_batch_labels = []  # Store true labels from each batch in this fold
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            # y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y_batch.cpu().numpy()

            fold_batch_probs.append(probs)
            fold_batch_labels.append(labels)

    # Concatenate all fold predictions
    y_scores_this_fold = np.concatenate(fold_batch_probs)
    y_true_this_fold = np.concatenate(fold_batch_labels)

    # Evaluate metrics
    roc_auc_fold = roc_auc_score(y_true_this_fold, y_scores_this_fold)
    precision_fold, recall_fold, f1_fold, _ = precision_recall_fscore_support(
        y_true_this_fold, y_scores_this_fold > 0.5, average="binary", zero_division=0
    )
    print(f"\n--- Performance for Fold {fold + 1} ---")
    print(f"ROC AUC:     {roc_auc_fold:.4f}")
    print(f"Precision:   {precision_fold:.4f}")
    print(f"Recall:      {recall_fold:.4f}")
    print(f"F1 Score:    {f1_fold:.4f}")

    # Append results for overall ensemble calculation
    overall_y_scores_accumulated.append(y_scores_this_fold)
    overall_y_labels_accumulated.append(y_true_this_fold)

# --- Overall Ensemble Performance Across All Folds ---
if not overall_y_labels_accumulated:
    print("\nNo validation results were accumulated. Cannot compute overall ensemble performance.")
else:
    y_true_overall = np.concatenate(overall_y_labels_accumulated)
    y_scores_overall = np.concatenate(overall_y_scores_accumulated)

    # Free up memory
    del overall_y_labels_accumulated
    del overall_y_scores_accumulated

    try:
        thresholds = np.arange(0.0, 1.0, 0.01)
        f1_scores = [f1_score(y_true_overall, (y_scores_overall >= t).astype(int), zero_division=0) for t in thresholds]
        
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1_score = f1_scores[best_threshold_idx]
        
        print("\n--- Optimal Threshold Search ---")
        print(f"Best threshold found: {best_threshold:.2f}")
        print(f"This threshold yields a maximum F1 score of: {best_f1_score:.4f}")

        # For P/R/F1, convert scores to binary predictions using the *optimal* threshold 
        y_preds_overall = (y_scores_overall >= best_threshold).astype(int)
        
        # Calculate final metrics using the optimal threshold
        roc_auc_overall = roc_auc_score(y_true_overall, y_scores_overall)
        accuracy_overall = accuracy_score(y_true_overall, y_preds_overall)
        precision_overall, recall_overall, f1_overall, _ = precision_recall_fscore_support(
            y_true_overall, y_preds_overall, average="binary", zero_division=0
        )

        # Calculate percentages
        num_total = len(y_preds_overall)
        num_pos = np.sum(y_preds_overall)
        num_neg = num_total - num_pos
        percent_pos = 100.0 * num_pos / num_total if num_total > 0 else 0
        percent_neg = 100.0 * num_neg / num_total if num_total > 0 else 0

        print("\n=== Overall Ensemble Performance (at Optimal Threshold) ===")
        print(f"ROC AUC:             {roc_auc_overall:.4f}")
        print(f"Accuracy:            {accuracy_overall:.4f}")
        print(f"Precision:           {precision_overall:.4f}")
        print(f"Recall:              {recall_overall:.4f}")
        print(f"F1 Score (verified): {f1_overall:.4f}")
        print(f"Predicted Positives: {num_pos} ({percent_pos:.2f}%)")
        print(f"Predicted Negatives: {num_neg} ({percent_neg:.2f}%)")

    except ValueError as e:
        print(f"Could not calculate overall ensemble metrics: {e}")
        print(f"Unique labels in overall val set: {np.unique(y_true_overall)}")

##### Focal loss?

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
