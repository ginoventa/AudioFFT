import numpy as np
import matplotlib.pyplot as plt

T = 20
N = 1000
x = np.linspace(-T/2, T/2, N, endpoint=False)
y = (x >= 0).astype(float)

f_fft = np.fft.fft(y) / N
magnitude = np.abs(f_fft)
frequencias = np.fft.fftfreq(N, d=T/N)

half_N = N // 2
f_pos = frequencias[:half_N]
m_pos = magnitude[:half_N]

plt.figure(figsize=(10, 5))
plt.plot(f_pos, m_pos, '*')
plt.title('Espectro do sinal degrau unitário')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
