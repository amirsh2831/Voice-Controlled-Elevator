import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

sample_rate, data = wav.read('traning_Samples/1-2.wav')
frame_size = 1024

averages = []
energies = []
zero_crossings = []

if len(data.shape) == 2:
    print("data.shape 2")
    data = np.mean(data, axis=1)

    emphasized = np.copy(data)
    emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

num_frames = (len(emphasized) - frame_size) // 512 + 1

for i in range(num_frames):

    start = i * 512
    end = start + frame_size
    
    frame = emphasized[start:end]
    avg = np.mean(frame)
    energy = np.sum(frame ** 2)
    zcr = 0

    for a in range(1, len(frame)):
        if (frame[a] * frame[a-1] < 0): 
            zcr += 1
        elif (frame[a] * frame[a-1] == 0):
            if (frame[a] * frame[a-2] < 0):
                zcr += 1

    averages.append(avg)
    energies.append(energy)
    zero_crossings.append(zcr)

frames = np.arange(num_frames)

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

axs[0].plot(frames, averages, label='Average', color='blue')
axs[0].set_ylabel('Average Amplitude')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(frames, energies, label='Energy', color='green')
axs[1].set_ylabel('Energy')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(frames, zero_crossings, label='Zero Crossings', color='red')
axs[2].set_xlabel('Frame Index')
axs[2].set_ylabel('Zero Crossings')
axs[2].legend()
axs[2].grid(True)

axs[3].plot( emphasized, label='main fram', color='orange')
axs[3].set_xlabel('Frame Index')
axs[3].set_ylabel('data')
axs[3].legend()
axs[3].grid(True)

plt.suptitle('elevator mk II')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()