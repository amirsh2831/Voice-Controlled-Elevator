import numpy as np
import scipy.io.wavfile as wav
import os
import csv

# Folder containing .wav files
folder_path = 'dataset'
frame_size = 1024

# Prepare CSV output
csv_file = 'voice_features.csv'
header = ['File Name', 'Mean Avg', 'Mean Energy', 'Mean ZCR']

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            try:
                sample_rate, data = wav.read(file_path)

                if len(data.shape) == 2:
                    data = np.mean(data, axis=1)

                num_frames = len(data) // frame_size
                averages = []
                energies = []
                zcrs = []

                for i in range(num_frames):
                    frame = data[i * frame_size : (i + 1) * frame_size]
                    if len(frame) == 0:
                        continue

                    avg = np.mean(frame)
                    energy = np.sum(frame ** 2)
                    zero_crossings = np.diff(np.sign(frame)) != 0
                    zcr = np.count_nonzero(zero_crossings)

                    averages.append(avg)
                    energies.append(energy)
                    zcrs.append(zcr)

                mean_avg = np.mean(averages)
                mean_energy = np.mean(energies)
                mean_zcr = np.mean(zcrs)

                writer.writerow([file_name, mean_avg, mean_energy, mean_zcr])

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
