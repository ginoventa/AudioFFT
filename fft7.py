import sounddevice as sd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

fs = 16000
duration = 5

print("Grave sua fala (5 segundos)...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
audio = audio.flatten()

N = len(audio)
yf = fft(audio)
xf = fftfreq(N, 1 / fs)

cutoff_freq = 500

filter_mask = (np.abs(xf) <= cutoff_freq)
yf_filtered = yf * filter_mask
audio_filtered = ifft(yf_filtered).real

audio_filtered /= np.max(np.abs(audio_filtered))
audio_filtered = audio_filtered.astype(np.float32)

print("Reproduzindo áudio filtrado...")
sd.play(audio_filtered, fs)
sd.wait()
print("Fim da reprodução.")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title("Espectro original")
plt.plot(xf[:N//2], np.abs(yf[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()

plt.subplot(2, 1, 2)
plt.title("Espectro filtrado (até 500 Hz)")
plt.plot(xf[:N//2], np.abs(yf_filtered[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()

plt.tight_layout()
plt.show()
