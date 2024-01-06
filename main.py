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
    axs[i].set_ylabel(r'U [$\mu V$]')
    axs[i].set_xlabel('t[s]')
    axs[i].set_title(f'EKG {i+1}')

plt.tight_layout()
plt.show()

ecg = numbers[4][6000::]
ecg_time = time[6000::]

plt.plot(ecg_time, ecg)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel(r'U [$\mu V$]')
plt.title('EKG signál')
plt.xlim([6, 10])
plt.show()


qrst = ecg[3060:3811]
qrst_time = ecg_time[3060:3811]
plt.plot(qrst_time, qrst)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel(r'U [$\mu V$]')
plt.title('EKG signál - PQRST komplex')
plt.show()

p = qrst[112:226]
pq = qrst[226:250]
q = qrst[250:275]
qr = qrst[275:305]
s = qrst[305:370]
st = qrst[370:485]
t = qrst[485:595]

p_time = qrst_time[112:226]
pq_time = qrst_time[226:250]
q_time = qrst_time[250:275]
qr_time = qrst_time[275:305]
s_time = qrst_time[305:370]
st_time = qrst_time[370:485]
t_time = qrst_time[485:595]

plt.plot(p_time, p, label = 'P')
plt.plot(pq_time, pq, label = 'PQ')
plt.plot(q_time, q, label = 'Q')
plt.plot(qr_time, qr, label = 'QR')
plt.scatter(9.363, 6820, label = 'R')
plt.plot(s_time, s, label = 'S')
plt.plot(st_time, st, label = 'ST')
plt.plot(t_time, t, label = 'T')
plt.grid()
plt.xlabel('t [s]')
plt.ylabel(r'U [$\mu V$]')
plt.title('EKG signál - PQRST komplex (detail)')
plt.legend()
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
print(b, a)
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
plt.xlabel('t [s]')
plt.ylabel(r'U [$\mu V$]')
plt.title('Původní signál')

plt.subplot(2, 1, 2)
plt.specgram(filtered_signal, Fs=fs, vmin=-20, vmax=50)
plt.xlabel('t [s]')
plt.ylabel(r'U [$\mu V$]')
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

print(len(qrst))
qrst = qrst[80::]
qrst_time = qrst_time[80::]
plt.title('ECG signál')
plt.legend()
plt.grid()
plt.show()
fft_result = np.fft.fft(qrst)
fft_freq = np.fft.fftfreq(len(qrst), d=1/fs)

# Calculate the power spectrum (magnitude of the FFT squared)
power_spectrum = np.abs(fft_result) ** 2

# Function for plotting
def plot_signal_spectrum():
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(qrst_time, qrst)
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
