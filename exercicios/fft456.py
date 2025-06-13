import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

duration = 5
fs = 16000

print("Grave sua nota (5 segundos)...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
audio = audio.flatten()

N = len(audio)
yf = fft(audio)
xf = fftfreq(N, 1 / fs)

idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])

plt.figure(figsize=(12, 6))
plt.plot(xf, yf)
plt.title("Espectro da gravação da nota cantada")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 5000)
plt.grid()
plt.show()
