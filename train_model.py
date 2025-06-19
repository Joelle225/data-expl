from sklearn.model_selection import KFold
from collections import defaultdict
from annotated_torch_dataset import SlidingWindowPoseDataset 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, f1_score
import torch 
import numpy as np

######ooo######
# Misc. Knobs
n_splits                = 5
neg_to_pos_ratio        = 1 # Balancing for training data
balance_dataset         = True # TODO reenable

# RF Hyperparameters
rf_n_estimators         = 1000
rf_max_depth            = None
rf_min_samples_split    = 8
rf_min_samples_leaf     = 4
rf_class_weight         = "balanced" # Handles imbalance within RF
######ooo######

# Load initial PyTorch sequence dataset
sequence_dataset_pt = torch.load("./extracted_drinking_sequence_dataset_features.pth", weights_only=False)

# Group sequences by participant-video-segment, not by camera and annotator, TODO try with this disabled?
# TODO alter dataset to serve windows new or smtn idfk
# TODO dynamic decision threshold in final calcs
# TODO maybe remove all null data?
grouped = defaultdict(list)
for idx, seq in enumerate(sequence_dataset_pt):
    meta = seq['meta']
    key = (meta['participant'], meta['video'], meta['segment'])
    grouped[key].append(idx)

group_keys = list(grouped.keys())
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store out-of-fold predictions and labels for overall ensemble evaluation
overall_y_scores_accumulated = [] # Probabilities for positive class
overall_y_labels_accumulated = []

for fold, (train_group_indices, val_group_indices) in enumerate(kf.split(group_keys)):
    print(f"\n=== Fold {fold + 1}/{n_splits} ===")

    train_indices = [idx for i in train_group_indices for idx in grouped[group_keys[i]]]
    val_indices = [idx for i in val_group_indices for idx in grouped[group_keys[i]]]

    train_sequences_pt = [sequence_dataset_pt[i] for i in train_indices]
    val_sequences_pt = [sequence_dataset_pt[i] for i in val_indices]

    # Create datasets for scikit-learn (will extract features)
    print("Loading and featurizing training set for scikit-learn...")
    train_dataset = SlidingWindowPoseDataset(
        sequences=train_sequences_pt,
        neg_to_pos_ratio=neg_to_pos_ratio,
        balance=balance_dataset,
        is_for_sklearn=True,
        seed=42 + fold
    )
    print(f"Training dataset size (sklearn): {len(train_dataset)}")

    print("Loading and featurizing validation set for scikit-learn...")
    val_dataset_sklearn = SlidingWindowPoseDataset(
        sequences=val_sequences_pt,
        balance=False,              # Evaluate on original (or differently balanced) validation distribution
        reverse_positives=False,    # No augmentation for validation
        neg_to_pos_ratio=2,
        is_for_sklearn=True,
        seed=100 + fold
    )
    
    print(f"Validation dataset size (sklearn): {len(val_dataset_sklearn)}")

    if len(train_dataset) == 0:
        print("Warning: Training dataset is empty for this fold. Skipping.")
        continue
    
    # Prepare data for scikit-learn's fit method
    # X_train will be a list of 1D feature arrays, y_train a list of labels
    # print(f"what is sample[1]? its: {train_dataset[1][1]}")
    X_train_list = [sample[0] for sample in train_dataset] # TODO is incorrect?, also check validation code
    y_train_list = [sample[1] for sample in train_dataset]
    
    if not X_train_list: # Should be caught by len(train_dataset_sklearn) == 0
        print("Warning: No training samples after processing. Skipping fold.")
        continue

    X_train_np = np.array(X_train_list)
    y_train_np = np.array(y_train_list)

    num_pos_train = np.sum(y_train_np == 1.0)
    num_neg_train = np.sum(y_train_np == 0.0)
    print(f"Num pos for RF training this fold: {num_pos_train}, Num neg: {num_neg_train}")

    if num_pos_train == 0 or num_neg_train == 0:
        print("Warning: Training data for RF has only one class. Skipping fold or using default class_weight.")

    # Initialize and train Random Forest
    model_rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        class_weight=rf_class_weight, # Handles imbalance
        random_state=42,
        n_jobs=-1 # Use all available cores
    )
    
    print("Training Random Forest model...")
    model_rf.fit(X_train_np, y_train_np)

    # --- Validation for THIS FOLD ---
    X_val_list = [sample[0] for sample in val_dataset_sklearn]
    y_val_list = [sample[1] for sample in val_dataset_sklearn]
    
    if not X_val_list:
        print("Warning: No validation samples after processing for this fold.")
    
    X_val_np = np.array(X_val_list)
    y_val_np = np.array(y_val_list)
    y_true_this_fold = np.array(y_val_list)

    num_pos_val = np.sum(y_val_np == 1.0)
    num_neg_val = np.sum(y_val_np == 0.0)
    print(f"Num pos for RF validation this fold: {num_pos_val}, Num neg: {num_neg_val}")

    y_probs_this_fold = model_rf.predict_proba(X_val_np)[:, 1] # Probabilities for the positive class
    y_preds_this_fold = model_rf.predict(X_val_np)

    try:
        roc_auc_fold = roc_auc_score(y_true_this_fold, y_probs_this_fold)
        accuracy_fold = accuracy_score(y_true_this_fold, y_preds_this_fold)
        precision_fold, recall_fold, f1_fold, _ = precision_recall_fscore_support(
            y_true_this_fold, y_preds_this_fold, average="binary", zero_division=0
        )
        # print(f"\n--- Performance for Fold {fold + 1} (Random Forest) ---")
        # print(f"ROC AUC:     {roc_auc_fold:.4f}")
        # print(f"Accuracy:    {accuracy_fold:.4f}")
        # print(f"Precision:   {precision_fold:.4f}")
        # print(f"Recall:      {recall_fold:.4f}")
        # print(f"F1 Score:    {f1_fold:.4f}")

        overall_y_scores_accumulated.append(y_probs_this_fold)
        overall_y_labels_accumulated.append(y_true_this_fold)
    except ValueError as e:
        print(f"Could not calculate metrics for fold {fold + 1}: {e}")
        print(f"Unique labels in this fold's val set: {np.unique(y_true_this_fold)}")

# --- Overall Ensemble Performance Across All Folds ---
if not overall_y_labels_accumulated:
    print("\nNo validation results were accumulated. Cannot compute overall ensemble performance.")
else:
    y_true_overall = np.concatenate(overall_y_labels_accumulated)
    y_scores_overall = np.concatenate(overall_y_scores_accumulated)

    # Free up memory
    del overall_y_labels_accumulated
    del overall_y_scores_accumulated, sequence_dataset_pt

    try:
        thresholds = np.arange(0.3, 0.8, 0.01)
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
