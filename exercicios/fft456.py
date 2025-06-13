import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

"""
Exercício 4 e 5 - Grave uma só nota e uma fala, e compare as diferenças entre eles.
"""

# Variáveis do programa
fs = 16000  # Definição do valor de frequência recomendada
duration = 5  # Definição da duração da gravação

# Gravação do áudio
print("Grave (5 segundos)...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
audio = audio.flatten()  # Transformação do áudio de 2D para 1D para facilitar a manipulação

# Análise do áudio gravado
N = len(audio)  # Número total de amostras do áudio gravado
yf = fft(audio)  # Aplica a Transformada Rápida de Fourier (FFT) ao sinal de áudio para obter o espectro de frequências
xf = fftfreq(N, 1 / fs)  # Gera o vetor de frequências correspondente a cada ponto do espectro obtido pela FFT

# Seleção das frequências positivas
idx = xf >= 0  # Máscara booleana para selecionar apenas as frequências positivas (maiores ou iguais a zero)
xf = xf[idx]  # Filtra o vetor de frequências para manter apenas as frequências positivas ou iguais a zero
yf = np.abs(yf[idx])  # Filtra o espectro de Fourier para manter apenas os valores correspondentes às frequências positivas e calcula o módulo (amplitude)

# Gráfico dos dados obtidos
plt.figure(figsize=(12, 6))
plt.plot(xf, yf)
plt.title("Espectro da gravação da nota cantada")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 5000)
plt.grid()
plt.show()
