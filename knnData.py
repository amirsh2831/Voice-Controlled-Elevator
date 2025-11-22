import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import ast

# === CONFIG ===
SAMPLEPATH = "traning_Samples/"   
DATASET_CSV = "dataset_knn.csv"
FRAME_SIZE = 1024
HOP_SIZE = 512
TARGET_FRAMES = 300  

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

all_features = []
frame_lengths = []

for filename in os.listdir(SAMPLEPATH):
    if filename.endswith('.wav'):
        path = os.path.join(SAMPLEPATH, filename)
        feats = extract_features(path)
        all_features.append(feats)
        frame_lengths.append(feats.shape[0])
        print(f"loaded {filename} with {feats.shape[0]} frames")



filenames = [f for f in os.listdir(SAMPLEPATH) if f.endswith('.wav')]

resized_features = []
for i, feat in enumerate(all_features):
    resized = resize_frames(feat, TARGET_FRAMES)

    avg_list = resized[:, 0].tolist()
    energy_list = resized[:, 1].tolist()
    zcr_list = resized[:, 2].tolist()
    filename = filenames[i].split("-")[0]
    print("sample", filename)
    resized_features.append([
        filename,
        avg_list,
        energy_list,
        zcr_list,
        TARGET_FRAMES
    ])

    print(f"Resized sample {filenames[i]} to {TARGET_FRAMES} frames")
    
with open(DATASET_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'average', 'energy', 'zcr', 'frame_count'])

    for row in resized_features:
        writer.writerow(row)



