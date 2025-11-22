import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import ast

NEW_SAMPLE_PATH = "test_files"   
DATASET_CSV = "dataset_knn.csv"
FRAME_SIZE = 1024
HOP_SIZE = 512
TARGET_FRAMES = 300  
CONFUSION_CSV = "confusion_matrix.csv"

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

    for i in range(w_start, w_end+ 1):
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

        # signs = np.sign(frame)
        # signs = signs[signs != 0]

        # if len(signs) > 1:
        #     zcr = np.mean(np.abs(np.diff(signs)))
        # else:
        #     zcr = 0

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

confusion_matrix = np.zeros((10, 10), dtype=int)

for test_file in os.listdir(NEW_SAMPLE_PATH):
    if not test_file.endswith(".wav"):
        continue

    path = os.path.join(NEW_SAMPLE_PATH, test_file)

    # ^ fix lables problem
    test_label = int(test_file.split('-')[0])  
    test_label_index = test_label - 1         

    features = extract(path)
    resized = resize(features, TARGET_FRAMES)
    resized = (resized - mean_feat) / std_feat

    distances = []
    for filename, sample_feat in dataset:
        dist = np.linalg.norm(resized - sample_feat)
        distances.append((filename, dist))

    distances.sort(key=lambda x: x[1])
    
    predicted_label_str = distances[0][0].split('.')[0] 
    predicted_label = int(predicted_label_str)
    predicted_label_index = predicted_label - 1 
    
    # ^ add data to confusion
    confusion_matrix[test_label_index][predicted_label_index] += 1

    print(f"Test: {test_file} | Actual: {test_label} | Predicted: {predicted_label}")
# === Save confusion matrix to CSV ===
with open(CONFUSION_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([""] + list(range(1, 11)))  # Header row
    for i in range(1, 11):
        writer.writerow([i] + list(confusion_matrix[i-1]))

# print(f"\n Confusion matrix saved to '{CONFUSION_CSV}'")