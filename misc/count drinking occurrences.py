import torch
from pathlib import Path
import numpy as np

# Data: Each entry is a dict: 
#   X: [T,17,2], Y: [T], meta: identifiers + frame list
#
# General shape of each entry -- dataset list where each element is:
# {
#   'X': torch.Tensor of shape [T,17,2],
#   'Y': torch.Tensor of shape [T],
#   'meta': {
#     'video':  ...,
#     'segment': ...,
#     'camera':  ...,
#     'participant': ...,
#     'frames': [list of ints]
#   }
# }

def count_drinking_occurrences(dataset, tolerance=0):
    lengths = []
    total_occurrences = 0

    for entry in dataset:
        labels = entry['Y'].numpy()  # [T] tensor → NumPy array
        count = 0
        i = 0
        while i < len(labels):
            # Skip zeros until a 1 is found
            if labels[i] == 1:
                # Start of a new drinking occurrence
                strt = i
                count += 1
                gap = 0
                i += 1
                while i < len(labels):
                    if labels[i] == 1:
                        gap = 0
                    else:
                        gap += 1
                        if gap > tolerance:
                            lengths.append(i - strt)
                            break # end of a drinking sequence
                    i += 1
            else:
                i += 1

        total_occurrences += count

    return total_occurrences, lengths

dataset = torch.load(Path("./drinking_sequence_dataset.pth"))
tolerance = 60
occurrences, lengths = count_drinking_occurrences(dataset, tolerance=tolerance)
print(f"Total drinking occurrences (tolerance={tolerance}): {occurrences}")
print(f"Mean {np.mean(lengths)}, Var: {np.var(lengths)}, len: {len(lengths)}")
