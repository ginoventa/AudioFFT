import numpy as np
import matplotlib.pyplot as plt

T = 20
N = 1000
x = np.linspace(-10, 10, N, endpoint=False)
y = (x >= 0).astype(float)

f_fft = np.fft.fft(y)
freqs = np.fft.fftfreq(N, d=T/N)

for k in range(20):
    print(f"f = {freqs[k]:.3f} Hz -> coef = {f_fft[k]:.3f}")
