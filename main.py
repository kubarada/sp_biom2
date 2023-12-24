import matplotlib.pyplot as plt
from scipy.signal import butter, freqz
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

# Filter specifications
fs = 360.0  # Sample rate, Hz
lowcut = 0.5  # Low cutoff frequency, Hz
highcut = 100.0  # High cutoff frequency, Hz

# Design a Butterworth bandpass filter using the butter function
b, a = butter(N=3, Wn=[lowcut/fs*2, highcut/fs*2], btype='band')  # N is the order of the filter

# Frequency response of the filter
w, h = freqz(b, a, worN=2000)

# Plot the frequency response.
plt.figure()
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(lowcut, 0.5*np.sqrt(2), 'ko')
plt.axvline(lowcut, color='k')
plt.plot(highcut, 0.5*np.sqrt(2), 'ko')
plt.axvline(highcut, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Bandpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()