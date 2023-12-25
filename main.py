import matplotlib.pyplot as plt
from scipy.signal import butter, freqz
from scipy.signal import lfilter
import numpy as np
import heartpy as hp


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


# filtering comparison
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

# spectrogram comparison
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.specgram(ecg, Fs=fs, vmin=-20, vmax=50)
plt.axhline(y=100, color='r', linestyle='--')
plt.xlabel('t [s]')
plt.ylabel('U [mV]')
plt.title('Původní signál')

plt.subplot(2, 1, 2)
plt.specgram(filtered_signal, Fs=fs, vmin=-20, vmax=50)
plt.axhline(y=100, color='r', linestyle='--')
plt.xlabel('t [s]')
plt.ylabel('U [mV]')
plt.title('Filtrovaný signál')

plt.tight_layout()
plt.show()


## determining heartrate
wd, m = hp.process(filtered_signal, sample_rate=fs)
# Get the calculated heart rate
heart_rate = m['bpm']  # bpm stands for beats per minute
print("Estimated Heart Rate:", heart_rate, "bpm")

# Plot the filtered ECG signal
plt.figure(figsize=(12, 4))
plt.plot(filtered_signal, label='Filtrované EKG', color='blue')

# Plot the detected R-peaks
plt.scatter(wd['peaklist'], [filtered_signal[j] for j in wd['peaklist']], color='red', label='R-maxima')

plt.title('ECG signál')
plt.legend()
plt.grid()
plt.show()

fft_result = np.fft.fft(filtered_signal)
fft_freq = np.fft.fftfreq(filtered_signal.size, d=1/fs)

# Calculate the power spectrum (magnitude of the FFT squared)
power_spectrum = np.abs(fft_result) ** 2

# Function for plotting
def plot_signal_spectrum():
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(ecg_time, filtered_signal)
    plt.title('EKG signál')
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(fft_freq, power_spectrum)
    plt.title('Výkonové spektrum')
    plt.grid()
    plt.xlabel('f [Hz]')
    plt.ylabel('Power')
    plt.xlim(0, 50)  # Limiting to show the relevant frequencies

    plt.tight_layout()
    plt.show()

# Call the function to plot the signal and its frequency power spectrum
plot_signal_spectrum()
