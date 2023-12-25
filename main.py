import matplotlib.pyplot as plt
from scipy.signal import butter, freqz
from scipy.signal import lfilter
import numpy as np

# Define the path to your file
file_path = 'input/rada.txt'

# Initialize an empty list to hold your numbers
numbers = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line_numbers = [int(num) for num in line.split()]
        numbers.append(line_numbers)

duration = 10
num_samples = len(numbers[0])
fs = num_samples/duration
print(fs)
time = [i / fs for i in range(len(numbers[0]))]



num_subplots = len(numbers)
fig, axs = plt.subplots(num_subplots)

if num_subplots == 1:
    axs = [axs]

for i, num_list in enumerate(numbers):
    axs[i].plot(time, num_list)
    axs[i].grid(True)
    axs[i].set_ylabel('U[mV]')
    axs[i].set_xlabel('t[s]')
    axs[i].set_title(f'EKG {i+1}')

plt.tight_layout()
plt.show()

ecg = numbers[4][6000::]
ecg_time = time[6000::]

plt.plot(ecg_time, ecg)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('U [mV]')
plt.title('EKG signál')
plt.xlim([6, 10])
plt.show()

# ploting spectogram
plt.figure(figsize=(15, 5))
plt.specgram(ecg, Fs=fs, vmin=-20, vmax=50)
plt.title('Spektrogram')
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.show()

fc = 100.0
filter_order = 2

Wn = fc/(0.5*fs)

b, a = butter(filter_order, Wn, btype='low', analog=False)
filtered_signal = lfilter(b, a, ecg)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_time, ecg, label='Původní signál')
plt.title('Původní signál')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ecg_time, filtered_signal, label='Filtrovaný signál', color='red')
plt.title('Filtrovaný signál')
plt.grid(True)

plt.tight_layout()
plt.show()
