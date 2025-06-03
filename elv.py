import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

sample_rate, data = wav.read('sara.wav')
frame_size = 1024
# num_frames = len(data) // frame_size

averages = []
energies = []
zero_crossings = []


# stereo to mono
if len(data.shape) == 2:
    print("data.shape 2")
    data = np.mean(data, axis=1)

    emphasized = np.copy(data)
    # emphasized[0] = data[0]  
    emphasized[1:] = 1.7 * (data[1:] - 0.99 * data[:-1])

# frams 50% overlapp
num_frames = (len(emphasized) - frame_size) // 512 + 1

for i in range(num_frames):

    # frame = data[i * frame_size : (i + 1) * frame_size]
    # frame = emphasized[i * frame_size : (i + 1) * frame_size]

    # 50 % overlap---------------
    start = i * 512
    end = start + frame_size
    
    frame = emphasized[start:end]
    # ---------------
    avg = np.mean(frame)
    energy = np.sum(frame ** 2)

    # --------------------
    signs = np.sign(frame)
    notzero = signs != 0
    zero_filter = signs[notzero]

    if len(zero_filter) > 1:
        zcr = np.mean(np.abs(np.diff(zero_filter)))
    else:
        zcr = 0
    # --------------------

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