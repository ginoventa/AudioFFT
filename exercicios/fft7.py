import sounddevice as sd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

"""
Exercício 7 - Filtre sua fala, retirando todos os componentes de frequência
            acima de 500 Hz. Reproduza sua voz. Para isso, obtenha o sinal através
            do polinômio interpolador Pn = a0 + ... + aN sin(2π1000x) + aN+1 cos(2π1000x),
            truncado na frequência que corresponde a mil Hz.
"""

# Definição dos parâmetros principais
fs = 16000           # Frequência de amostragem recomendada (Hz)
duration = 5         # Duração da gravação em segundos
cutoff_freq = 500    # Frequência de corte para o filtro (Hz)

# Gravação do áudio
print("Grave sua fala (5 segundos)...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)  # Gravação do áudio mono
sd.wait()
audio = audio.flatten()  # Transformação do áudio de 2D para 1D para facilitar manipulações

# Cálculo de parâmetros básicos
N = len(audio)  # Número total de amostras gravadas
t = np.linspace(0, duration, N, endpoint=False)  # Vetor de tempo correspondente ao sinal

# Transformada Rápida de Fourier para análise espectral
yf = fft(audio)               # FFT do sinal original
xf = fftfreq(N, 1 / fs)       # Vetor de frequências associado ao espectro

#
# FILTRAGEM POR MEIO DO TRUNCAMENTO DIRETO DO FFT
#
filter_mask = (np.abs(xf) <= cutoff_freq)  # Máscara booleana para manter só até 500 Hz
yf_fft_filtered = yf * filter_mask         # Aplicação da máscara ao espectro
audio_fft_filtered = ifft(yf_fft_filtered).real  # Transformada inversa (reconstrução do sinal)
audio_fft_filtered /= np.max(np.abs(audio_fft_filtered))  # Normalização do áudio filtrado
audio_fft_filtered = audio_fft_filtered.astype(np.float32)  # Conversão para tipo adequado à reprodução

#
# FILTRAGEM PELO POLINÔMIO INTERPOLADOR
#
# Seleção das componentes positivas até 500 Hz
idx_pos = np.where((xf >= 0) & (xf <= cutoff_freq))[0]     # Índices das frequências válidas
frequencias = xf[idx_pos]                                 # Frequências até 500 Hz
coeficientes = yf[idx_pos] / N                            # Coeficientes normalizados

# Reconstrução do sinal usando senos e cossenos (polinômio truncado)
Pn = np.zeros_like(t)  # Inicialização do vetor reconstruído
for i, f in enumerate(frequencias):
    a_k = 2 * np.real(coeficientes[i])                    # Coeficiente do cosseno
    b_k = -2 * np.imag(coeficientes[i])                   # Coeficiente do seno (sinal invertido para correção de fase)
    Pn += a_k * np.cos(2 * np.pi * f * t) + b_k * np.sin(2 * np.pi * f * t)

# Normalização e formatação do sinal reconstruído
Pn /= np.max(np.abs(Pn))
Pn = Pn.astype(np.float32)
Pn *= np.sqrt(np.sum(audio_fft_filtered**2)) / np.sqrt(np.sum(Pn**2))

#Reprodução do aúdio após ambas as filtragens
print("Reproduzindo áudio filtrado via truncamento FFT...")
sd.play(audio_fft_filtered, fs)
sd.wait()

print("Reproduzindo áudio reconstruído com polinômio interpolador...")
sd.play(Pn, fs)
sd.wait()
print("Fim da reprodução.")

#Gráficos da faixa de áudio original, filtrada por FFT e filtrada pelo polinômio interpolador

plt.figure(figsize=(14, 10))

# Espectro original
plt.subplot(3, 1, 1)
plt.title("Espectro Original")
plt.plot(xf[:N//2], np.abs(yf[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()

# Espectro do sinal filtrado por FFT
plt.subplot(3, 1, 2)
plt.title("Espectro após filtragem por truncamento na FFT")
plt.plot(xf[:N//2], np.abs(yf_fft_filtered[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()

# Espectro do sinal reconstruído via interpolador
plt.subplot(3, 1, 3)
plt.title("Espectro do sinal reconstruído (polinômio até 500 Hz)")
yf_pn = fft(Pn)
plt.plot(xf[:N//2], np.abs(yf_pn[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()

plt.tight_layout()
plt.show()
