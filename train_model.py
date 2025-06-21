from sklearn.model_selection import KFold
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from annotated_torch_dataset import SlidingWindowPoseDataset
from cnn import DrinkingCNN, train_model
import torch
# from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, roc_curve, precision_recall_curve, average_precision_score, auc
import datetime
import matplotlib.pyplot as plt
from draw_linegraph import *

############
### TODO's
## *1. array size too big
## *2. step size to erratic/big
## *3. feature extraction
## *4. TQDM in data init
## 5. refactor into neater code
## *6. fix leak in train/val split on annotator as well
## -7. with this little data, should I even use a CNN?
## -8. add noise to the positives and re-add them into the sampler
# TODO try gigantic window size

## Attention: TODO check if use correct: PyTorch's Conv1d typically expects (batch_size, channels, sequence_length), so (batch_size, N, W) if N is your number of feature channels
############

# Store metrics from each fold
all_roc_aucs = []
all_precisions = []
all_recalls = []
all_f1s = []

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
val_neg_to_pos_ratio    = 10               
val_balance_dataset     = True

batch_size              = 32
bce_pos_weight_factor   = 4 # was 350
num_epochs              = 20
######ooo######

# Misc Options
save_model_weights=False
plot_y_values = True
plot_2_curves = False
plot_pr_vs_threshold = False

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

# lists to store data for the new plot
all_fold_precisions_vs_thresh = []
all_fold_recalls_vs_thresh = []

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
        reverse_positives=False,
        percentage_pos=0.3
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # Model, optimizer, loss
    sample_X, _, _ = train_dataset[0]
    input_channel_size = sample_X.shape[1] * sample_X.shape[2]
    model = DrinkingCNN(input_channels=input_channel_size).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if len(train_dataset) > 0:
        train_labels_for_weight = [sample_tuple[1].item() for sample_tuple in train_dataset]
        num_pos_train = sum(1 for label in train_labels_for_weight if label == 1.0)
        num_neg_train = len(train_labels_for_weight) - num_pos_train

        val_labels_for_weight = [sample_tuple[1].item() for sample_tuple in val_dataset] 
        num_pos_val = sum(1 for label in val_labels_for_weight if label == 1.0)
        num_neg_val = len(val_labels_for_weight) - num_pos_val

        print(f"Num pos this fold: {num_pos_train}, Num negative this fold: {num_neg_train}")
        print(f"Num pos val this fold: {num_pos_val}, Num negative val this fold: {num_neg_val}")

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
    if save_model_weights:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), f"cnn_model_fold{fold+1}_{timestamp}.pth")

    # --- Find the optimal threshold on the TRAINING data ---
    model.eval()
    train_probs_for_threshold = []
    train_labels_for_threshold = []
    with torch.no_grad():
        # Note: Because of the lack of data we compromise and use training data to determine optimal f1 so there is no leakage (shouldn't do this on val set)
        for X_batch, y_batch, met_batch in train_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            train_probs_for_threshold.append(probs)
            train_labels_for_threshold.append(y_batch.cpu().numpy())

    train_probs_for_threshold = np.concatenate(train_probs_for_threshold)
    train_labels_for_threshold = np.concatenate(train_labels_for_threshold)

    thresholds = np.arange(0.0, 0.95, 0.01)
    f1_scores = [f1_score(train_labels_for_threshold, (train_probs_for_threshold >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold for this fold (from training data): {best_threshold:.2f}")


    # Generate a plot for each group
    if plot_y_values:
        ######### plotting ytrue vs ypred #########
        print("\n--- Evaluating and generating plots for validation set ---")
        model.eval()
        fold_results = []
        with torch.no_grad():
            # Iterate through the validation loader to get predictions with metadata
            for X_batch, y_batch, meta_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                # Apply sigmoid to get probabilities, then flatten
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                labels = y_batch.cpu().numpy().flatten()

                # Store each prediction with its corresponding metadata
                for i in range(len(probs)):
                    fold_results.append({
                        'prob': probs[i],
                        'label': labels[i],
                        'meta': meta_batch[i]
                    })

        # Group results by video segment to create separate plots
        grouped_for_plotting = defaultdict(list)
        for result in fold_results:
            meta = result['meta']
            key = (meta['participant'], meta['video'], meta['segment'], meta['camera'])
            grouped_for_plotting[key].append(result)

        # Create a directory for the fold's plots if it doesn't exist
        plot_dir = f"./fold_{fold+1}_plots"
        os.makedirs(plot_dir, exist_ok=True)

        for key, results in grouped_for_plotting.items():
            participant, video, segment, camera = key

            # Sort results by the start frame to ensure the time axis is correct
            results.sort(key=lambda r: r['meta']['start_frame'])

            frames = [r['meta']['start_frame'] for r in results]
            preds = [r['prob'] for r in results]
            truths = [r['label'] for r in results]

            plt.figure(figsize=(18, 6))
            # Plot ground truth as a stepped line to show clear label boundaries
            plt.step(frames, truths, where='post', label='Ground Truth Label', color='green', linestyle='--', linewidth=2)
            # Plot model predictions
            plt.plot(frames, preds, label='Model Prediction (Probability)', color='blue', alpha=0.8, marker='.', markersize=4)
            # Plot the determined threshold as a reference line
            plt.axhline(y=best_threshold, color='red', linestyle=':', label=f'Optimal F1 Threshold ({best_threshold:.2f})')

            plt.title(f"Fold {fold+1}: Predictions vs. Truth\nParticipant {participant} | Video {video} | Segment {segment} | Camera {camera}")
            plt.xlabel("Start Frame of Window")
            plt.ylabel("Label / Probability")
            plt.ylim(-0.1, 1.1)
            plt.legend(loc='upper left')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Sanitize the filename
            filename = f"p{participant}_v{video}_s{segment}_c{camera}.png"
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path)
            plt.close() # Close the figure to free up memory

        print(f"Generated {len(grouped_for_plotting)} plots in the '{plot_dir}' directory.\n")

    # --- Evaluate on the VALIDATION data using the determined threshold ---
    y_scores_this_fold = []
    y_true_this_fold = []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_scores_this_fold.append(probs)
            y_true_this_fold.append(y_batch.cpu().numpy())

    y_scores_this_fold = np.concatenate(y_scores_this_fold)
    y_true_this_fold = np.concatenate(y_true_this_fold)

    # Now, use the best_threshold to calculate metrics for this fold
    y_pred_this_fold = (y_scores_this_fold >= best_threshold).astype(int)

    roc_auc_fold = roc_auc_score(y_true_this_fold, y_scores_this_fold)
    precision_fold, recall_fold, f1_fold, _ = precision_recall_fscore_support(
        y_true_this_fold, y_pred_this_fold, average="binary", zero_division=0
    )

    if not np.isnan(roc_auc_fold): all_roc_aucs.append(roc_auc_fold)
    else : print("Warn: NaN roc_auc detected!")
    all_precisions.append(precision_fold)
    all_recalls.append(recall_fold)
    all_f1s.append(f1_fold)

    print(f"\n--- Performance for Fold {fold + 1} ---")
    print(f"ROC AUC:     {roc_auc_fold:.4f}")
    print(f"Precision:   {precision_fold:.4f}")
    print(f"Recall:      {recall_fold:.4f}")
    print(f"F1 Score:    {f1_fold:.4f}")

    # Append results for overall ensemble calculation
    # It's better to store predictions and evaluate at the end.
    if not plot_y_values: overall_y_scores_accumulated.append(y_scores_this_fold)
    if not plot_y_values: overall_y_labels_accumulated.append(y_true_this_fold)

    # --- Generate Precision-Recall vs. Threshold plot for this fold --- #
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


######## plotting ########
if plot_2_curves:
    #################################################################
    # 1. Plotting the Receiver Operating Characteristic (ROC) Curve #
    #################################################################
    plt.figure(figsize=(10, 8))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Plot ROC curve for each fold
    for i in range(n_splits):
        fpr, tpr, thresholds = roc_curve(overall_y_labels_accumulated[i], overall_y_scores_accumulated[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Fold {i + 1} (AUC = {roc_auc:.2f})')
        
        # Interpolate TPRs at mean_fpr points
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    # Plot the random guesser line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guesser', alpha=.8)

    # Plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})',
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Final plot settings
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"roc_curves{datetime.datetime.now()}.png")


    ###############################################################
    # 2. Plotting the Precision-Recall (PR) Curve                 #
    ###############################################################

    plt.figure(figsize=(10, 8))

    # Concatenate all fold results to calculate the baseline
    all_y_true = np.concatenate(overall_y_labels_accumulated)
    pos_proportion = np.sum(all_y_true) / len(all_y_true)

    # Plot the random guesser line for PR curve
    plt.plot([0, 1], [pos_proportion, pos_proportion], linestyle='--', lw=2, color='r', 
            label=f'Random Guesser (AP = {pos_proportion:.2f})', alpha=.8)

    # Plot PR curve for each fold
    for i in range(n_splits):
        precision, recall, _ = precision_recall_curve(overall_y_labels_accumulated[i], overall_y_scores_accumulated[i])
        avg_precision = average_precision_score(overall_y_labels_accumulated[i], overall_y_scores_accumulated[i])
        plt.plot(recall, precision, lw=1, alpha=0.5,
                label=f'Fold {i + 1} (AP = {avg_precision:.2f})')

    # Final plot settings
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"prec-rec_curves{datetime.datetime.now()}.png")

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
