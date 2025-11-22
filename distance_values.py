import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import ast

NEW_SAMPLE_PATH = "test_files/2-10.wav"   
DATASET_CSV = "dataset_knn.csv"
FRAME_SIZE = 1024
HOP_SIZE = 512
TARGET_FRAMES = 300  

def resize(data , target_frames): 
    original_frames, features = data.shape
    resized = np.zeros((target_frames, features))
    scale = (original_frames - 1) / (target_frames - 1)

    for i in range(target_frames - 1):
        src_index = i * scale
        j = int(src_index)
        wx = src_index - j
        if j + 1 < original_frames:
            resized[i] = data[j] * (1 - wx) + data[j + 1] * wx
        else:
            resized[i] = data[j]
    resized[-1] = data[-1]
    return resized

def extract(filepath):
    sample_rate, data = wav.read(filepath)
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    emphasized = np.copy(data)
    emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

    num_frames = (len(emphasized) - FRAME_SIZE) // HOP_SIZE + 1
     # ^ apply VAD to the signal
    first_energies = []
    for e in range(num_frames):
        start = e * HOP_SIZE
        end = start + FRAME_SIZE
        frame = emphasized[start:end]

        energy = np.sum(frame  ** 2)
        first_energies.append(energy)
    
    normalized_energies = first_energies / max(first_energies)
    w_start = 0
    w_end = 0

    for ws in range(num_frames): 
        if (normalized_energies[ws] > 0.03): 
            w_start = ws
            break

    for we in range(num_frames - 1, -1, -1): 
        if (normalized_energies[we] > 0.03): 
            w_end = we
            break

    features = []

    for i in range(w_start, w_end + 1):
        start = i * HOP_SIZE
        end = start + FRAME_SIZE
        frame = emphasized[start:end]

        if len(frame) < FRAME_SIZE:
            continue

        avg = np.mean(frame)
        energy = np.sum(frame ** 2)
        zcr = 0

        for a in range(1, len(frame)):
         if (frame[a] * frame[a-1] < 0): 
            zcr += 1
         elif (frame[a] * frame[a-1] == 0):
            if (frame[a] * frame[a-2] < 0):
                zcr += 1

        features.append([avg, energy, zcr])

    return np.array(features)

dataset = []
with open(DATASET_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename']
        avg = np.array(ast.literal_eval(row['average']))
        energy = np.array(ast.literal_eval(row['energy']))
        zcr = np.array(ast.literal_eval(row['zcr']))
        combined = np.stack([avg, energy, zcr], axis=1)
        dataset.append((filename, combined))

all_feat = np.stack([f for _, f in dataset])                    # (N, 300, 3)
mean_feat = all_feat.mean(axis=(0,1), keepdims=True)
std_feat = all_feat.std(axis=(0,1), keepdims=True) + 1e-8

# Normalize all templates
for i in range(len(dataset)):
    name, feat = dataset[i]
    dataset[i] = (name, (feat - mean_feat) / std_feat)

features = extract(NEW_SAMPLE_PATH)
features_resized = resize(features, TARGET_FRAMES)
features_resized = (features_resized - mean_feat) / std_feat

distances = []
for filename, sample_feat in dataset:
    dist = np.linalg.norm(features_resized - sample_feat)
    distances.append((filename, dist))

distances.sort(key=lambda x: x[1])
print("\nSimilarity Ranking:")
for fname, d in distances:
    print(f"{fname:30} Distance: {d:.2f}")

minimums = distances[:3]
min_labels = [fname.split('-')[0] for fname, _ in minimums]

from collections import Counter
label_counts = Counter(min_labels)
most_common_label, count = label_counts.most_common(1)[0]

print("\n" + "="*50)
print("Top 3 closest matches:")
for i, (fname, dist) in enumerate(minimums):
    label = fname.split('-')[0].lstrip('0')
    print(f"  {i+1}. {fname:20} → Distance: {dist:.3f} (label: {label})")

print(f"\n predicted: {most_common_label}")
# print(f"→ Confidence: {count}/3 votes")
if count == 3:
    print("   PERFECT MATCH!")
elif count == 2:
    print("   Strong prediction")
else:
    print("   Weak prediction")
print("="*50)

# if count >= 2:
#     print(f"Predicted Label: {most_common_label}")
# else:
#     print("Prediction: Unknown (no majority)")