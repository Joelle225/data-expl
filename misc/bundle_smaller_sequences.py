import torch
from pathlib import Path
from collections import defaultdict

# Load saved dataset
dataset_path = Path("./drinking_sequence_dataset.pth")
dataset = torch.load(dataset_path)  # list of dicts with 'X', 'Y', 'meta'

# Count unique participants
participants = set()
for entry in dataset:
    meta = entry['meta']
    participant_id = (meta['participant'])
    participants.add(participant_id)

print(f"Found {len(participants)} unique participants with annotated data.")

# The following was vybe coded:

### Counts label imbalance when uncommented:
# label_counts = [0, 0]
# for entry in dataset:
#     Ys = entry["Y"]
#     for y in Ys:
#         label_counts[int(y)] += 1

# print(f"label imbalance: {label_counts[0]} 0's found, and {label_counts[1]} 1's found.")

# # --- Step 2: Split sequences into chunks of 45 frames ---
# chunk_len = 45
# chunked_dataset = []

# for entry in dataset:
#     X, Y, meta = entry['X'], entry['Y'], entry['meta']
#     T = X.shape[0]

#     for start in range(0, T - chunk_len + 1, chunk_len):
#         end = start + chunk_len
#         X_chunk = X[start:end] # [45, 17, 2]
#         Y_chunk = Y[start:end] # [45,]

#         chunk_meta = meta.copy()
#         chunk_meta['frames'] = meta['frames'][start:end]  # preserve corresponding frame indices
#         chunk_meta['chunk_start'] = start
#         chunk_meta['chunk_end'] = end

#         chunked_dataset.append({
#             'X': X_chunk,
#             'Y': Y_chunk,
#             'meta': chunk_meta
#         })

# print(f"Split into {len(chunked_dataset)} sequences of {chunk_len} frames.")
