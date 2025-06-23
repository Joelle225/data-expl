import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    sequence_dataset_pt = torch.load("./drinking_sequence_dataset.pth")
except FileNotFoundError:
    print("Error: The file './drinking_sequence_dataset.pth' was not found.")
    print("Please ensure the dataset file is in the correct directory.")
    exit()

KEYPOINT_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8',
    9: '9', 10: '10', 11: '11', 12: '12',
    13: '13', 14: '14', 15: '15', 16: '16'
}

for KEYPOINT_OF_INTEREST_IDX in range(len(KEYPOINT_MAP)):
    keypoint_name = KEYPOINT_MAP.get(KEYPOINT_OF_INTEREST_IDX, f'Keypoint {KEYPOINT_OF_INTEREST_IDX}')

    # --- 1. Calculate Features and Flatten Data ---
    # We will create a list of records for each frame, then convert it to a DataFrame.
    records = []
    for sequence in sequence_dataset_pt:
        # Convert tensors to numpy arrays for easier processing
        X = sequence['X'].numpy()
        Y = sequence['Y'].numpy()

        # Calculate velocity as the displacement between consecutive frames.
        # The shape of 'velocity' will be [T-1, 17, 2].
        velocity = np.diff(X, axis=0)

        # Calculate speed as the Euclidean norm (magnitude) of the velocity vector.
        # The shape of 'speed' will be [T-1, 17].
        speed = np.linalg.norm(velocity, axis=2)

        # The labels in Y correspond to each of the T frames. Since speed is calculated
        # between frames (t and t+1), it has T-1 entries. We align them by assigning
        # the label from frame t+1 to the movement that ended at t+1.
        Y_aligned = Y[1:]

        # Store the speed of the keypoint of interest and its corresponding label for each frame.
        for frame_idx in range(speed.shape[0]):
            records.append({
                'label': Y_aligned[frame_idx],
                'speed': speed[frame_idx, KEYPOINT_OF_INTEREST_IDX]
            })

    df = pd.DataFrame(records)

    # --- 2. Visualize the Correlation ---
    # Set a style for the plots
    sns.set_style("whitegrid")

    # Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='speed', data=df, palette="Set2")
    plt.title(f'Distribution of Speed for "{keypoint_name}" vs. Label', fontsize=16)
    plt.xlabel('Action Label', fontsize=12)
    plt.ylabel('Speed (pixels per frame)', fontsize=12)
    # plt.show()
    plt.savefig(f"boxplot_speed_vs_label_{KEYPOINT_OF_INTEREST_IDX}.png")

    # Create a violin plot for a more detailed view of the distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='label', y='speed', data=df, palette="Set3", inner='quartile')
    plt.title(f'Distribution of Speed for "{keypoint_name}" vs. Label', fontsize=16)
    plt.xlabel('Action Label', fontsize=12)
    plt.ylabel('Speed (pixels per frame)', fontsize=12)
    # plt.show() 
    plt.savefig(f"violin_speed_vs_label_{KEYPOINT_OF_INTEREST_IDX}.png")
