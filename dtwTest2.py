import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

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

# ---------------------------------------------------
sample_rate, data = wav.read('sara.wav')
frame_size = 1024
overlap = 512  
if len(data.shape) == 2:
    data = np.mean(data, axis=1)

emphasized = np.copy(data)
emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

# ---------------------------------------------------

num_frames = (len(emphasized) - frame_size) // overlap + 1
averages, energies, zero_crossings = [], [], []

for i in range(num_frames):
    start = i * overlap
    end = start + frame_size
    frame = emphasized[start:end]

    avg = np.mean(frame)
    energy = np.sum(frame ** 2)

    signs = np.sign(frame)
    notzero = signs != 0
    zero_filter = signs[notzero]

    if len(zero_filter) > 1:
        zcr = np.mean(np.abs(np.diff(zero_filter)))
    else:
        zcr = 0

    averages.append(avg)
    energies.append(energy)
    zero_crossings.append(zcr)

features = np.column_stack((averages, energies, zero_crossings))
TARGET_FRAMES = 150 
resized_features = resize_frames(features, TARGET_FRAMES)
frames = np.arange(TARGET_FRAMES)

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

axs[0].plot(frames, resized_features[:,0], label='Average', color='blue')
axs[0].set_ylabel('Average Amplitude')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(frames, resized_features[:,1], label='Energy', color='green')
axs[1].set_ylabel('Energy')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(frames, resized_features[:,2], label='Zero Crossings', color='red')
axs[2].set_xlabel('Frame Index')
axs[2].set_ylabel('Zero Crossings')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(emphasized, label='main fram', color='orange')
axs[3].set_xlabel('Frame Index')
axs[3].set_ylabel('data')
axs[3].legend()
axs[3].grid(True)

plt.suptitle('elevator mk II')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()