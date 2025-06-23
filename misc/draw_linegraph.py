import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# This is a placeholder for your actual dataset class.
# Ensure its __getitem__ returns (X, Y, meta)
from annotated_torch_dataset import SlidingWindowPoseDataset


def get_detailed_predictions(model, val_sequences, window_size, device, best_threshold):
    """
    Runs the model on validation sequences with a stride of 1 to get
    detailed predictions for every possible window.
    """
    print("Generating detailed predictions for plotting...")
    # 1. Create a special dataset for evaluation with stride=1
    eval_dataset = SlidingWindowPoseDataset(
        sequences=val_sequences,
        window_size=window_size,
        stride=1,  # Use a stride of 1 for full resolution
        balance=False,  # Use all data
        reverse_positives=False # Don't augment during evaluation
    )

    if len(eval_dataset) == 0:
        print("Warning: Evaluation dataset is empty. Skipping detailed predictions.")
        return [], [], []

    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 2. Get model predictions for every window
    model.eval()
    all_probs = []
    all_labels = []
    all_metas = []

    with torch.no_grad():
        for X_batch, y_batch, meta_batch in tqdm(eval_loader, desc="Predicting for plots"):
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            # The 'meta' from the dataloader is a list of dicts. We need to parse it.
            # We restructure it to be a dict of lists.
            restructured_meta = defaultdict(list)
            for i in range(len(meta_batch['Y'])):
                for key in meta_batch:
                    # Some items might not be lists (e.g., a single tensor), handle them.
                    if torch.is_tensor(meta_batch[key]):
                        restructured_meta[key].append(meta_batch[key][i].item() if meta_batch[key].numel() > 1 else meta_batch[key].item())
                    elif isinstance(meta_batch[key], list):
                         restructured_meta[key].append(meta_batch[key][i])
                    else: # Handle non-list/tensor items if necessary
                        restructured_meta[key].append(meta_batch[key])


            for i in range(probs.shape[0]):
                all_probs.append(probs[i])
                all_labels.append(y_batch[i].item())
                # Reconstruct individual meta dictionary for each sample
                sample_meta = {key: restructured_meta[key][i] for key in restructured_meta}
                all_metas.append(sample_meta)

    # Convert probabilities to binary predictions
    all_preds = (np.array(all_probs) >= best_threshold).astype(int)

    return all_preds, all_labels, all_metas


def process_and_plot_time_series(all_preds, all_metas, sequence_dataset, output_dir="plots", use_majority_vote=False):
    """
    Processes detailed predictions, aggregates them to the frame level,
    and generates time series plots.
    """
    if not all_preds:
        print("No predictions to process for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Aggregating predictions and generating plots (saved to '{output_dir}')...")

    # --- Step 1: Group predictions by video segment and camera ---
    # Key: (participant, video, segment, camera)
    # Value: Dict containing predictions and metadata for that group
    grouped_results = defaultdict(list)
    for pred, meta in zip(all_preds, all_metas):
        key = (meta['participant'], meta['video'], meta['segment'], meta['camera'])
        grouped_results[key].append({'pred': pred, 'meta': meta})

    # --- Step 2: Process each group to generate plots ---
    for group_key, results in tqdm(grouped_results.items(), desc="Generating Plots"):
        participant, video_name, segment, camera = group_key

        # Find the original full sequence to get frame-by-frame ground truth
        original_sequence = None
        for seq in sequence_dataset:
            meta = seq['meta']
            if (meta['participant'] == participant and
                meta['video'] == video_name and
                meta['segment'] == segment):
                # We can have multiple annotators, so we group by annotator later.
                # Just grab the first matching sequence to get the total frame count.
                original_sequence = seq
                break
        
        if original_sequence is None:
            continue

        num_frames = len(original_sequence['frames'])
        frame_predictions = [[] for _ in range(num_frames)]

        # --- Step 3: Map window predictions back to individual frames ---
        for res in results:
            start_idx = res['meta']['start_idx']
            end_idx = start_idx + res['meta']['X'].shape[0] # Use the actual window size
            for i in range(start_idx, end_idx):
                if i < num_frames:
                    frame_predictions[i].append(res['pred'])
        
        # --- Step 4: Aggregate frame predictions using the chosen strategy ---
        y_pred_aggregated = np.zeros(num_frames, dtype=int)
        for i in range(num_frames):
            preds_for_frame = frame_predictions[i]
            if not preds_for_frame:
                continue
            
            if use_majority_vote:
                # Majority vote: 1 if >50% of predictions are 1
                y_pred_aggregated[i] = 1 if np.mean(preds_for_frame) > 0.5 else 0
            else:
                # Any vote: 1 if any prediction is 1
                y_pred_aggregated[i] = 1 if np.sum(preds_for_frame) > 0 else 0
        
        # --- Step 5: Get ground truth for all annotators for this segment ---
        annotator_gts = defaultdict(lambda: np.zeros(num_frames, dtype=int))
        for seq in sequence_dataset:
             meta = seq['meta']
             if (meta['participant'] == participant and
                 meta['video'] == video_name and
                 meta['segment'] == segment and
                 meta['camera'] == camera):
                 annotator = meta.get('annotator', 'unknown')
                 annotator_gts[annotator] = seq['Y'].numpy()

        # --- Step 6: Check if the plot is worth creating ---
        has_positive_example = np.any(y_pred_aggregated == 1)
        if not has_positive_example:
            for gt in annotator_gts.values():
                if np.any(gt == 1):
                    has_positive_example = True
                    break
        
        if not has_positive_example:
            continue # Skip plotting if all labels and predictions are 0

        # --- Step 7: Create the multi-panel plot ---
        num_annotators = len(annotator_gts)
        fig, axes = plt.subplots(num_annotators, 1, figsize=(20, 4 * num_annotators), sharex=True, squeeze=False)
        fig.suptitle(f'Participant {participant} - Video {video_name} - Segment {segment} - Cam {camera}\nAggregation: {"Majority Vote" if use_majority_vote else "Any Vote"}', fontsize=16)

        for i, (annotator, y_true) in enumerate(annotator_gts.items()):
            ax = axes[i, 0]
            frames = np.arange(num_frames)
            
            # Plot Ground Truth
            ax.plot(frames, y_true, label=f'Ground Truth (Annotator: {annotator})', color='green', drawstyle='steps-post')
            
            # Plot Aggregated Prediction
            ax.plot(frames, y_pred_aggregated, label='Model Prediction', color='orange', drawstyle='steps-post', alpha=0.8)
            
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Not Drinking', 'Drinking'])
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        axes[-1, 0].set_xlabel('Frame Index within Segment')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_filename = f"{participant}_{video_name}_seg{segment}_cam{camera}.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close(fig)

