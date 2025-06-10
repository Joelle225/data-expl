import torch
from pathlib import Path

def count_drinking_occurrences(dataset, tolerance=0):
    total_occurrences = 0

    for entry in dataset:
        labels = entry['Y'].numpy()  # [T] tensor → NumPy array
        count = 0
        i = 0
        while i < len(labels):
            # Skip zeros until a 1 is found
            if labels[i] == 1:
                # Start of a new drinking occurrence
                count += 1
                gap = 0
                i += 1
                while i < len(labels):
                    if labels[i] == 1:
                        gap = 0
                    else:
                        gap += 1
                        if gap > tolerance:
                            break
                    i += 1
            else:
                i += 1

        total_occurrences += count

    return total_occurrences

dataset = torch.load(Path("./drinking_sequence_dataset.pth"))
tolerance = 60
occurrences = count_drinking_occurrences(dataset, tolerance=tolerance)
print(f"Total drinking occurrences (tolerance={tolerance}): {occurrences}")
