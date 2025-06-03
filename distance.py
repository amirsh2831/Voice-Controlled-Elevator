import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import ast

# === CONFIG ===
NEW_SAMPLE_PATH = "10-test.wav"   
DATASET_CSV = "dataset.csv"
FRAME_SIZE = 1024
HOP_SIZE = 512
TARGET_FRAMES = 131  

def resize_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
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

def extract_features(filepath):
    sample_rate, data = wav.read(filepath)
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    emphasized = np.copy(data)
    emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

    num_frames = (len(emphasized) - FRAME_SIZE) // HOP_SIZE + 1
    features = []

    for i in range(num_frames):
        start = i * HOP_SIZE
        end = start + FRAME_SIZE
        frame = emphasized[start:end]

        if len(frame) < FRAME_SIZE:
            continue

        avg = np.mean(frame)
        energy = np.sum(frame ** 2)
        signs = np.sign(frame)
        signs = signs[signs != 0]

        if len(signs) > 1:
            zcr = np.mean(np.abs(np.diff(signs)))
        else:
            zcr = 0

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

features = extract_features(NEW_SAMPLE_PATH)
features_resized = resize_frames(features, TARGET_FRAMES)

distances = []
for filename, sample_feat in dataset:
    dist = np.linalg.norm(features_resized - sample_feat)
    distances.append((filename, dist))

distances.sort(key=lambda x: x[1])
print("\nSimilarity Ranking:")
for fname, d in distances:
    print(f"{fname:30} Distance: {d:.2f}")

names = [x[0] for x in distances]
values = [x[1] for x in distances]

plt.figure(figsize=(12, 6))
plt.bar(names, values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Euclidean Distance")
plt.title("Distance from New Sample to Dataset Samples")
plt.tight_layout()
plt.show()
