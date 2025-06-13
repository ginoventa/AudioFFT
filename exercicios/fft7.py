import sounddevice as sd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

##
#Exercício 7 - Filtre sua fala, retirando todos componentes de frequência
#              acima de 500 Hz. Reproduza sua voz. Para isso, obtenha o sinal através
#              do polinômio interpolador Pn = a0+· · ·+aN sin(2π1000x)+aN+1 cos(2π1000x),
#              truncado na frequência que corresponde a mil Hz.
##

#Variáveis do programa
fs = 16000 #Definição do valor de frequência recomendada
duration = 5 #Definição da duração da gravação
cutoff_freq = 500 #Frequência máxima

#Gravação do aúdio
print("Grave sua fala (5 segundos)...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
audio = audio.flatten() #Flatten do aúdio obtido, transformação de 2D para 1D a fim de facilitar a manipulação]

#Estudo matemático do aúdio gravado
N = len(audio) #Calcula o número total de amostras do áudio gravado
yf = fft(audio) #Aplica a Transformada Rápida de Fourier (FFT) ao sinal de áudio para obter o espectro de frequências
xf = fftfreq(N, 1 / fs) #Gera o vetor de frequências correspondentes a cada ponto do espectro obtido pela FFT

#Filtragem do aúdio para Hz acima de 500 
filter_mask = (np.abs(xf) <= cutoff_freq)
yf_filtered = yf * filter_mask
audio_filtered = ifft(yf_filtered).real
audio_filtered /= np.max(np.abs(audio_filtered))
audio_filtered = audio_filtered.astype(np.float32)

#Reprodução do aúdio após a filtragem
print("Reproduzindo áudio filtrado...")
sd.play(audio_filtered, fs)
sd.wait()
print("Fim da reprodução.")

#Plot do gráfico antes e após a filtragem
plt.figure(figsize=(12, 8))
#Gráfico antes da filtragem
plt.subplot(2, 1, 1)
plt.title("Espectro original")
plt.plot(xf[:N//2], np.abs(yf[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()
#Gráfico após da filtragem
plt.subplot(2, 1, 2)
plt.title("Espectro filtrado (até 500 Hz)")
plt.plot(xf[:N//2], np.abs(yf_filtered[:N//2]))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid()

plt.tight_layout()
plt.show()
