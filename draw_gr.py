import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import datetime
import os
import matplotlib.pyplot as plt

# Assuming your model and training loop functions are in 'cnn.py'
from cnn import DrinkingCNN, train_model
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score

# --- Utility Functions ---

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of (X, Y, meta) tuples.
    It stacks tensors and gathers metadata into a list.
    """
    # batch is a list of tuples: [(X1, Y1, meta1), (X2, Y2, meta2), ...]
    X_list = [item[0] for item in batch]
    Y_list = [item[1] for item in batch]
    meta_list = [item[2] for item in batch]

    # Stack the tensors
    X_batch = torch.stack(X_list)
    Y_batch = torch.stack(Y_list)

    return X_batch, Y_batch, meta_list


class SlidingWindowPoseDataset(Dataset):
    """
    Patched SlidingWindowPoseDataset.
    The __getitem__ method now returns a tensor for the label and includes metadata.
    """
    def __init__(
        self,
        sequences,
        window_size=45,
        stride=1,
        neg_to_pos_ratio=3,
        balance=True,
        jitter_max=5,
        reverse_positives=True,
        seed=42,
        percentage_pos=0.6
    ):
        self.window_size = window_size
        self.stride = stride
        self.jitter_max = jitter_max
        self.reverse_positives = reverse_positives
        self.balance = balance
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.samples = []
        self.percentage_pos=percentage_pos

        random.seed(seed)
        np.random.seed(seed)

        pos_samples = []
        neg_samples = []

        for seq in tqdm(sequences, desc="Initializing Dataset"):
            X = seq['X']   # [T, 17, 2]
            Y = seq['Y']   # [T]
            meta = seq['meta']
            T = X.shape[0]

            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                Y_win = Y[start:end]
                label_val = (Y_win.float().mean() >= percentage_pos).float().item()

                if label_val == 0 and torch.any(Y_win > 0):
                    continue

                select_keypoints = [1, 2, 3, 5, 6, 8, 9, 12] # head, shoulders, hands
                newX = X[start:end][:, select_keypoints, :]
                sample = {
                    'X': newX,
                    'Y': label_val,
                    'meta': {
                        **meta,
                        'start_frame': meta['frames'][start],
                        'end_frame': meta['frames'][end - 1],
                        'start_idx': start
                    }
                }

                if label_val == 1.0:
                    pos_samples.append(sample)
                    if reverse_positives:
                        rev_sample = {
                            'X': torch.flip(newX, dims=[0]),
                            'Y': label_val,
                            'meta': {**sample['meta'], 'reversed': True}
                        }
                        pos_samples.append(rev_sample)
                else:
                    neg_samples.append(sample)

        if balance:
            keep_n_neg = min(len(neg_samples), len(pos_samples) * neg_to_pos_ratio)
            random.shuffle(neg_samples)
            self.samples = pos_samples + neg_samples[:keep_n_neg]
        else:
            self.samples = pos_samples + neg_samples

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = sample['X'].nan_to_num(nan=1.0)
        # Return label as a tensor
        Y = torch.tensor(sample['Y'], dtype=torch.float32)
        meta = sample['meta']
        return X, Y, meta

# --- Plotting Functions ---

def get_detailed_predictions(model, val_sequences, window_size, device, best_threshold):
    print("\nGenerating detailed predictions for plotting...")
    eval_dataset = SlidingWindowPoseDataset(
        sequences=val_sequences, window_size=window_size, stride=1,
        balance=False, reverse_positives=False
    )

    if len(eval_dataset) == 0:
        print("Warning: Evaluation dataset is empty.")
        return [], [], []

    # IMPORTANT FIX: Use the custom collate function
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    model.eval()
    all_probs, all_labels, all_metas = [], [], []

    with torch.no_grad():
        # IMPORTANT FIX: Handle meta_list correctly
        for X_batch, y_batch, meta_list in tqdm(eval_loader, desc="Predicting for plots"):
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.flatten())
            all_labels.extend(y_batch.cpu().numpy())
            all_metas.extend(meta_list)

    all_preds = (np.array(all_probs) >= best_threshold).astype(int)
    return all_preds, all_labels, all_metas


def process_and_plot_time_series(all_preds, all_metas, sequence_dataset, output_dir="plots", use_majority_vote=False):
    if not all_metas:
        print("No predictions to process for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Aggregating predictions and generating plots (saved to '{output_dir}')...")

    grouped_results = defaultdict(list)
    for pred, meta in zip(all_preds, all_metas):
        key = (meta['participant'], meta['video'], meta['segment'], meta['camera'])
        grouped_results[key].append({'pred': pred, 'meta': meta})

    for group_key, results in tqdm(grouped_results.items(), desc="Generating Plots"):
        participant, video_name, segment, camera = group_key

        original_sequence = next((s for s in sequence_dataset if s['meta']['participant'] == participant and s['meta']['video'] == video_name and s['meta']['segment'] == segment), None)
        if not original_sequence: continue

        num_frames = len(original_sequence['frames'])
        frame_predictions = [[] for _ in range(num_frames)]

        for res in results:
            start_idx = res['meta']['start_idx']
            end_idx = start_idx + train_window_size # Use the fixed window size
            for i in range(start_idx, end_idx):
                if i < num_frames:
                    frame_predictions[i].append(res['pred'])
        
        y_pred_aggregated = np.zeros(num_frames, dtype=int)
        for i in range(num_frames):
            preds_for_frame = frame_predictions[i]
            if not preds_for_frame: continue
            if use_majority_vote:
                y_pred_aggregated[i] = 1 if np.mean(preds_for_frame) > 0.5 else 0
            else:
                y_pred_aggregated[i] = 1 if np.any(preds_for_frame) else 0

        annotator_gts = defaultdict(lambda: np.zeros(num_frames, dtype=int))
        for seq in sequence_dataset:
             meta = seq['meta']
             if (meta['participant'] == participant and meta['video'] == video_name and meta['segment'] == segment and meta['camera'] == camera):
                 annotator = meta.get('annotator', 'unknown')
                 annotator_gts[annotator] = seq['Y'].numpy()

        has_positive_example = np.any(y_pred_aggregated == 1) or any(np.any(gt == 1) for gt in annotator_gts.values())
        if not has_positive_example: continue

        num_annotators = len(annotator_gts)
        fig, axes = plt.subplots(num_annotators, 1, figsize=(20, 4 * num_annotators), sharex=True, squeeze=False)
        fig.suptitle(f'P{participant} - V{video_name} - S{segment} - Cam {camera}\nAggregation: {"Majority Vote" if use_majority_vote else "Any Vote"}', fontsize=16)

        for i, (annotator, y_true) in enumerate(annotator_gts.items()):
            ax = axes[i, 0]
            frames = np.arange(num_frames)
            ax.plot(frames, y_true, label=f'GT (Annotator: {annotator})', color='green', drawstyle='steps-post')
            ax.plot(frames, y_pred_aggregated, label='Prediction', color='orange', drawstyle='steps-post', alpha=0.8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(['No', 'Yes']); ax.set_ylim(-0.1, 1.1)
            ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.7)

        axes[-1, 0].set_xlabel('Frame Index within Segment')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"P{participant}_V{video_name}_S{segment}_C{camera}.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close(fig)

# --- Main Training Script ---

if __name__ == "__main__":
    # Knobs to turn:
    n_splits = 5
    train_window_size = 180
    train_stride = 60
    train_neg_to_pos_ratio = 2
    train_balance_dataset = True
    train_reverse_positives = False
    learning_rate = 3e-4
    val_window_size = train_window_size
    batch_size = 32
    bce_pos_weight_factor = 4
    num_epochs = 20
    save_model_weights = False

    # Load dataset
    sequence_dataset = torch.load("./drinking_sequence_dataset.pth")
    grouped = defaultdict(list)
    for idx, seq in enumerate(sequence_dataset):
        key = (seq['meta']['participant'], seq['meta']['video'], seq['meta']['segment'])
        grouped[key].append(idx)

    group_keys = list(grouped.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_roc_aucs, all_precisions, all_recalls, all_f1s = [], [], [], []
    overall_y_scores_accumulated, overall_y_labels_accumulated = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(group_keys)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        train_indices = [idx for i in train_idx for idx in grouped[group_keys[i]]]
        val_indices = [idx for i in val_idx for idx in grouped[group_keys[i]]]
        train_sequences = [sequence_dataset[i] for i in train_indices]
        val_sequences = [sequence_dataset[i] for i in val_indices]

        print("Loading training set")
        train_dataset = SlidingWindowPoseDataset(sequences=train_sequences, window_size=train_window_size, stride=train_stride, neg_to_pos_ratio=train_neg_to_pos_ratio, balance=train_balance_dataset, reverse_positives=train_reverse_positives)
        print("Loading validation set")
        val_dataset = SlidingWindowPoseDataset(sequences=val_sequences, window_size=val_window_size, stride=5, neg_to_pos_ratio=4, balance=False)

        # IMPORTANT FIX: Use the custom collate function for all DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        
        sample_X, _, _ = train_dataset[0] # IMPORTANT FIX: Unpack 3 items
        input_channel_size = sample_X.shape[1] * sample_X.shape[2]
        model = DrinkingCNN(input_channels=input_channel_size).to(device)
        
        train_labels = [s['Y'] for s in train_dataset.samples]
        num_pos_train = sum(1 for label in train_labels if label == 1.0)
        num_neg_train = len(train_labels) - num_pos_train
        
        if num_pos_train > 0:
            effective_pos_weight = torch.tensor(bce_pos_weight_factor * (num_neg_train / num_pos_train), device=device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=effective_pos_weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs)

        if save_model_weights:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # torch.save(model.state_dict(), f"cnn_model_fold{fold+1}_{timestamp}.pth")

        # Find optimal threshold on training data
        train_probs, train_labels = [], []
        with torch.no_grad():
            # IMPORTANT FIX: Unpack 3 items from loader
            for X_batch, y_batch, _ in train_loader:
                logits = model(X_batch.to(device))
                train_probs.append(torch.sigmoid(logits).cpu().numpy())
                train_labels.append(y_batch.cpu().numpy())

        train_probs = np.concatenate(train_probs)
        train_labels = np.concatenate(train_labels)
        thresholds = np.arange(0.0, 1.0, 0.01)
        f1_scores = [f1_score(train_labels, (train_probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Best threshold for this fold: {best_threshold:.2f}")

        # Evaluate on validation data
        y_scores, y_true = [], []
        with torch.no_grad():
            # IMPORTANT FIX: Unpack 3 items from loader
            for X_batch, y_batch, _ in val_loader:
                logits = model(X_batch.to(device))
                y_scores.append(torch.sigmoid(logits).cpu().numpy())
                y_true.append(y_batch.cpu().numpy())
        
        y_scores = np.concatenate(y_scores)
        y_true = np.concatenate(y_true)
        y_pred = (y_scores >= best_threshold).astype(int)
        
        # --- Metrics and Plotting ---
        roc_auc_fold = roc_auc_score(y_true, y_scores)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        all_roc_aucs.append(roc_auc_fold); all_precisions.append(precision); all_recalls.append(recall); all_f1s.append(f1)
        overall_y_scores_accumulated.append(y_scores.flatten()); overall_y_labels_accumulated.append(y_true.flatten())

        # Generate time series plots
        fold_preds, _, fold_metas = get_detailed_predictions(model, val_sequences, train_window_size, device, best_threshold)
        output_plot_dir = f"./plots_fold_{fold+1}"
        process_and_plot_time_series(fold_preds, fold_metas, sequence_dataset, f"{output_plot_dir}/any_vote", use_majority_vote=False)
        process_and_plot_time_series(fold_preds, fold_metas, sequence_dataset, f"{output_plot_dir}/majority_vote", use_majority_vote=True)

    # --- Overall Performance and Final Plots ---
    print("\n--- Overall Cross-Validation Performance ---")
    print(f"Average ROC AUC: {np.mean(all_roc_aucs):.4f} (+/- {np.std(all_roc_aucs):.4f})")
