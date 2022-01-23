import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (9, 7)

sampFreq, sound = wavfile.read("/Users/cbalingbing/Desktop/2022 Test/22 Jan/recording45.wav")
sound.dtype, sampFreq
sound = sound / 2.0**15
sound.shape
length_in_s = sound.shape[0] / sampFreq
print(length_in_s)

plt.subplot(2,1,1)
plt.plot(sound[:,0], 'r')
plt.title("Pressure values and time", loc = 'left')
plt.xlabel("left channel, record at 45% boost")
plt.subplot(2,1,2)
plt.plot(sound[:,1], 'b')
plt.xlabel("right channel, record at 45% boost")
plt.tight_layout()
plt.show()

time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
plt.subplot(2,1,1)
plt.plot(time, sound[:,0], 'r')
plt.title("Signal versus time", loc = 'left')
plt.xlabel("time, s [left channel]")
plt.ylabel("signal, relative units")
plt.subplot(2,1,2)
plt.plot(time, sound[:,1], 'b')
plt.xlabel("time, s [right channel]")
plt.ylabel("signal, relative units")
plt.tight_layout()
plt.show()

signal = sound[:,0]
plt.plot(time[6000:7000], signal[6000:7000])
plt.title("Signal with higher resolution against time", loc = 'left')
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()

fft_spectrum = np.fft.rfft(signal)
freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
fft_spectrum
fft_spectrum_abs = np.abs(fft_spectrum)
plt.plot(freq, fft_spectrum_abs)
plt.title("Plot of Amplitude and frequency")
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

plt.plot(freq[:3000], fft_spectrum_abs[:3000])
plt.title("Plot of highest peak")
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

plt.plot(freq[:500], fft_spectrum_abs[:500])
plt.title("Zoom in on peaks")
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.arrow(90, 5500, -20, 1000, width=2, head_width=8, head_length=200, fc='k', ec='k')
plt.arrow(200, 4000, 20, -1000, width=2, head_width=8, head_length=200, fc='g', ec='g')
plt.show()