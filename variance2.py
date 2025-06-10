import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration and Data Loading ---
# Use a non-interactive backend for saving files
import matplotlib
matplotlib.use('Agg')

# A clear mapping for keypoints to be used in the final plot
KEYPOINT_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8',
    9: '9', 10: '10', 11: '11', 12: '12',
    13: '13', 14: '14', 15: '15', 16: '16'
}
NUM_KEYPOINTS = len(KEYPOINT_MAP)

try:
    # Load the dataset from the .pth file
    sequence_dataset_pt = torch.load("./drinking_sequence_dataset.pth")
except FileNotFoundError:
    print("Error: The file './drinking_sequence_dataset.pth' was not found.")
    print("Please ensure the dataset file is in the correct directory.")
    exit()

# --- Step 1: Process all sequences to gather variance data ---
# This list will store the variance results from each individual sequence
variance_records = []

for sequence in sequence_dataset_pt:
    # Convert tensors to numpy arrays for processing
    X = sequence['X'].numpy()  # Shape: [T, 17, 2]
    Y = sequence['Y'].numpy()  # Shape: [T]

    # --- Find boundaries to create variable-length chunks ---
    # Find indices where the label changes from 0 to 1 or 1 to 0
    changes = np.where(np.diff(Y) != 0)[0] + 1
    
    # Define the start and end points of each chunk
    boundaries = sorted(list(set([0] + list(changes) + [len(Y)])))

    drinking_chunks_X = []
    non_drinking_chunks_X = []

    # --- Extract chunks based on boundaries ---
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        # Skip empty chunks that might result from consecutive boundaries
        if start >= end:
            continue
        
        # We label the chunk based on the label of its first frame
        chunk_label = Y[start]
        chunk_X = X[start:end]
        
        if chunk_label == 1:
            drinking_chunks_X.append(chunk_X)
        else:
            non_drinking_chunks_X.append(chunk_X)
    
    # --- Calculate variances for the concatenated chunks within this sequence ---
    # For all drinking chunks in this sequence
    if drinking_chunks_X:
        drinking_concat = np.concatenate(drinking_chunks_X, axis=0)
        # Variance of x-coordinates (axis 0 is time, axis 1 is keypoints, axis 2 is coords)
        drinking_var_x = np.var(drinking_concat[:, :, 0], axis=0) # Shape: [17]
        # Variance of y-coordinates
        drinking_var_y = np.var(drinking_concat[:, :, 1], axis=0) # Shape: [17]
    else:
        # If no drinking chunks, variance is zero
        drinking_var_x = np.zeros(NUM_KEYPOINTS)
        drinking_var_y = np.zeros(NUM_KEYPOINTS)

    # For all non-drinking chunks in this sequence
    if non_drinking_chunks_X:
        non_drinking_concat = np.concatenate(non_drinking_chunks_X, axis=0)
        non_drinking_var_x = np.var(non_drinking_concat[:, :, 0], axis=0)
        non_drinking_var_y = np.var(non_drinking_concat[:, :, 1], axis=0)
    else:
        # If no non-drinking chunks, variance is zero
        non_drinking_var_x = np.zeros(NUM_KEYPOINTS)
        non_drinking_var_y = np.zeros(NUM_KEYPOINTS)

    # Store the calculated variances for this sequence
    variance_records.append({
        'drinking_var_x': drinking_var_x, 'drinking_var_y': drinking_var_y,
        'non_drinking_var_x': non_drinking_var_x, 'non_drinking_var_y': non_drinking_var_y,
    })

# --- Step 2: Average the variances across all sequences ---
# We take the mean of the variances calculated from each sequence
avg_drinking_var_x = np.mean([r['drinking_var_x'] for r in variance_records], axis=0)
avg_drinking_var_y = np.mean([r['drinking_var_y'] for r in variance_records], axis=0)
avg_non_drinking_var_x = np.mean([r['non_drinking_var_x'] for r in variance_records], axis=0)
avg_non_drinking_var_y = np.mean([r['non_drinking_var_y'] for r in variance_records], axis=0)

# --- Step 3: Calculate the difference in variance ---
# This is the value we actually want to plot
variance_diff_x = avg_drinking_var_x - avg_non_drinking_var_x
variance_diff_y = avg_drinking_var_y - avg_non_drinking_var_y

# --- Step 4: Plot the results in a single, combined bar graph ---
keypoint_labels = [KEYPOINT_MAP[i] for i in range(NUM_KEYPOINTS)]
x_indices = np.arange(NUM_KEYPOINTS)  # The label locations for the keypoints
bar_width = 0.35  # The width of the bars

fig, ax = plt.subplots(figsize=(18, 8))

# Create the bars for the X and Y variance differences
rects1 = ax.bar(x_indices - bar_width/2, variance_diff_x, bar_width, label='X Variance Difference (Drinking - Non-Drinking)')
rects2 = ax.bar(x_indices + bar_width/2, variance_diff_y, bar_width, label='Y Variance Difference (Drinking - Non-Drinking)')

# Add labels, title, and legend
ax.set_ylabel('Difference in Position Variance', fontsize=14)
ax.set_title('Increased Movement Variance During Drinking Actions by Keypoint', fontsize=16)
ax.set_xticks(x_indices)
ax.set_xticklabels(keypoint_labels, rotation=45, ha="right")
ax.legend()

# Add a horizontal line at y=0 to easily see positive vs. negative differences
ax.axhline(0, color='grey', linewidth=0.8)

# Use tight_layout to ensure labels don't overlap
fig.tight_layout()

# Save the single figure and show it
output_filename = 'variance_difference_by_keypoint.png'
plt.savefig(output_filename)
plt.show()

print(f"Plot saved to {output_filename}")
