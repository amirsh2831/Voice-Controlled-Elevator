import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# === Interpolation-based frame resizing ===
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

# === Load and process ===
sample_rate, data = wav.read('sara.wav')
frame_size = 1024
hop_size = 512  # 50% overlap

# Convert stereo to mono
if len(data.shape) == 2:
    data = np.mean(data, axis=1)

# Pre-emphasis filter
emphasized = np.copy(data)
emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

# === Frame-based feature extraction ===
num_frames = (len(emphasized) - frame_size) // hop_size + 1
averages, energies, zero_crossings = [], [], []

for i in range(num_frames):
    start = i * hop_size
    end = start + frame_size
    frame = emphasized[start:end]

    avg = np.mean(frame)
    energy = np.sum(frame ** 2)

    # Zero Crossing Rate (ZCR)
    signs = np.sign(frame)
    signs = signs[signs != 0]  # remove zeros
    if len(signs) > 1:
        zcr = np.mean(np.abs(np.diff(signs)))
    else:
        zcr = 0

    averages.append(avg)
    energies.append(energy)
    zero_crossings.append(zcr)

# === Combine features ===
features = np.column_stack((averages, energies, zero_crossings))
print(f"Original frame count: {features.shape[0]}")

# === Resize to fixed number of frames ===
TARGET_FRAMES = 250  # Change this based on average frame count across all your files
resized_features = resize_frames(features, TARGET_FRAMES)
print(f"Resized shape: {resized_features.shape}")

# === Plot for inspection ===
frames = np.arange(TARGET_FRAMES)
labels = ['Average', 'Energy', 'ZCR']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(frames, resized_features[:, i], label=labels[i], color=colors[i])
plt.legend()
plt.title("Interpolated Features to Fixed Length")
plt.xlabel("Frame Index")
plt.grid(True)
plt.tight_layout()
plt.show()

