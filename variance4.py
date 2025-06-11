import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYPOINT_MAP = {
    0: 'head', 1: 'nose', 2: 'neck', 3: 'rShoulder', 4: 'rElbow',
    5: 'rWrist', 6: 'lShoulder', 7: 'lElbow', 8: 'lWrist',
    9: 'rHip', 10: 'rKnee', 11: 'rAnkle', 12: 'lHip',
    13: 'lKnee', 14: 'lAnkle', 15: 'rFoot', 16: 'lFoot'
}

def calc_variance_corrected(window):
    """
    Calculates the variance for each keypoint after robustly handling NaNs.
    """
    window_x = window[:, :, 0]
    window_y = window[:, :, 1]
    if np.isnan(window_x).any():
        mean_per_keypoint_x = np.nanmean(window_x, axis=0)
        nan_indices_x = np.where(np.isnan(window_x))
        window_x[nan_indices_x] = np.take(mean_per_keypoint_x, nan_indices_x[1])
    if np.isnan(window_y).any():
        mean_per_keypoint_y = np.nanmean(window_y, axis=0)
        nan_indices_y = np.where(np.isnan(window_y))
        window_y[nan_indices_y] = np.take(mean_per_keypoint_y, nan_indices_y[1])
    cleaned_window = np.stack([window_x, window_y], axis=2)
    return np.var(cleaned_window, axis=0)

try:
    sequence_dataset_pt = torch.load("./drinking_sequence_dataset.pth")
except FileNotFoundError:
    print("Error: The file './drinking_sequence_dataset.pth' was not found.")
    exit()

window_size = 30
all_window_data = []

# --- Data Collection Loop ---
for seq in sequence_dataset_pt:
    X_full_seq = seq['X']
    Y_full_seq = seq['Y']
    T = X_full_seq.shape[0]

    for start in range(0, T - window_size + 1, window_size):
        end = start + window_size
        
        X_window_raw = X_full_seq[start:end]
        Y_win_labels = Y_full_seq[start:end]
        
        num_positive_frames = torch.sum(Y_win_labels > 0).item()
        
        data_x = X_window_raw.numpy()
        variances = calc_variance_corrected(data_x) # Shape [17, 2]
        
        # Calculate the average of X and Y variance for each keypoint
        avg_variances = np.mean(variances, axis=1) # Shape [17,]
        
        # Store data for this window
        window_dict = {'num_positive_frames': num_positive_frames}
        for i in range(17):
            keypoint_name = KEYPOINT_MAP[i]
            window_dict[f'{keypoint_name}_avg_var'] = avg_variances[i]
        
        all_window_data.append(window_dict)

# Convert the collected data to a pandas DataFrame
df = pd.DataFrame(all_window_data)

# --- Plotting Section ---
fig, axes = plt.subplots(3, 6, figsize=(20, 10))
axes = axes.flatten()

for k in range(17):
    ax = axes[k]
    keypoint_name = KEYPOINT_MAP[k]
    
    # Define the columns for the scatter plot
    x_col = 'num_positive_frames'
    y_col = f'{keypoint_name}_avg_var'
    
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        ax=ax,
        alpha=0.4,
        s=15, # marker size
        edgecolor=None # remove marker borders for cleanliness
    )
    
    ax.set_title(keypoint_name)
    ax.set_yscale('log')
    # We can remove individual axis labels to de-clutter the plot
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Hide the last unused subplot
axes[17].axis('off')

# Add common labels and a title to the entire figure
fig.supxlabel('Number of "Drinking" Frames in Window', fontsize=14)
fig.supylabel('Average Variance (X & Y, log scale)', fontsize=14)
fig.suptitle('Keypoint Variance vs. Number of Positive Frames', fontsize=20, y=1.03)

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
plt.savefig('variance_scatter_grid.png', bbox_inches='tight')
plt.close()

print("17-subplot scatter grid saved to 'variance_scatter_grid.png'")