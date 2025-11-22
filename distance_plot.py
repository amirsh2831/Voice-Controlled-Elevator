import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import ast

NEW_SAMPLE_PATH = "04-test.wav"   
DATASET_CSV = "dataset_knn.csv"
FRAME_SIZE = 1024
HOP_SIZE = 512
TARGET_FRAMES = 300  

def resize(data, target_frames):
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

features = extract(NEW_SAMPLE_PATH)
features_resized = resize(features, TARGET_FRAMES)

distances = []
for filename, features in dataset:
    dist = np.linalg.norm(features_resized - features)
    distances.append((filename, dist))

distances.sort(key=lambda x: x[0])

print("\nSimilarity Ranking:")
for fname, d in distances:
    print(f"{fname:30} Distance: {d:.2f}")

x_labels = [fname.split('-')[0] for fname, _ in distances]  
y_values = [dist for _, dist in distances]

x_label = list(range(len(x_labels)))

# Plot
plt.figure(figsize=(14, 6))
plt.plot(x_label, y_values, marker='o', linestyle='-')
plt.xticks(x_label, x_labels, rotation=45)
plt.xlabel("sample")
plt.ylabel("distance")
plt.title("test file distance to each sample")
plt.grid(True)
plt.tight_layout()
plt.show()
