import os
import numpy as np
import scipy.io.wavfile as wav
import csv

# === CONFIG ===
AUDIO_FOLDER = "dataset"  
OUTPUT_CSV = "dataset.csv"
FRAME_SIZE = 1024
OVERLAP = 512  



def resize_frames(data, target_frames):
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

    # Convert stereo to mono
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    emphasized = np.copy(data)
    emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

    num_frames = (len(emphasized) - FRAME_SIZE) // OVERLAP + 1
    features = []

    for i in range(num_frames):
        start = i * OVERLAP
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

all_features = []
frame_lengths = []

for filename in os.listdir(AUDIO_FOLDER):
    if filename.endswith('.wav'):
        path = os.path.join(AUDIO_FOLDER, filename)
        feats = extract(path)
        all_features.append(feats)
        frame_lengths.append(feats.shape[0])
        print(f"loaded {filename} with {feats.shape[0]} frames")

avg_frames = int(round(sum(frame_lengths) / len(frame_lengths)))
print(f"\naverage frame: {avg_frames}")

resized_features = []

filenames = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]

for i, feat in enumerate(all_features):
    resized = resize_frames(feat, avg_frames)

    avg_list = resized[:, 0].tolist()
    energy_list = resized[:, 1].tolist()
    zcr_list = resized[:, 2].tolist()

    resized_features.append([
        filenames[i],
        avg_list,
        energy_list,
        zcr_list,
        avg_frames
    ])

    print(f"Resized sample {filenames[i]} to {avg_frames} frames")

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'average', 'energy', 'zcr', 'frame_count'])

    for row in resized_features:
        writer.writerow(row)


# 
# dataset = []

# for filename in os.listdir(AUDIO_FOLDER):
#     if filename.endswith('.wav'):
#         path = os.path.join(AUDIO_FOLDER, filename)
#         feats = extract_features(path)

#         # if feats.size == 0:
#         #     continue  

#         frame_count = feats.shape[0]
#         avg_avg = np.mean(feats[:, 0])
#         avg_energy = np.mean(feats[:, 1])
#         avg_zcr = np.mean(feats[:, 2])

#         dataset.append([filename, avg_avg, avg_energy, avg_zcr, frame_count])
#         print(f"Processed {filename}: {frame_count} frames")


# with open(OUTPUT_CSV, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['filename', 'average', 'energy', 'zcr', 'frame_count'])  # header
#     writer.writerows(dataset)