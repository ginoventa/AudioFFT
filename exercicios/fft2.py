import numpy as np
import matplotlib.pyplot as plt

##
#Exercício 2 - Frequências relacionadas aos valores de FFT
##

#Variáveis do programa
T = 20 #Intervalo
N = 1000 #Número de amostras
x = np.linspace(-10, 10, N, endpoint=False) #Definição da malha espaçada por intervalos de 0.02, com exceção de 10
y = (x >= 0).astype(float) #Definição dos valores unitários

#Cálculo do FFT e da frequência
f_fft = np.fft.fft(y) #FFT
freqs = np.fft.fftfreq(N, d=T/N) #Frequências

#Print dos 20 resultados obtidos
for k in range(20):
    print(f"f = {freqs[k]:.3f} Hz -> coef = {f_fft[k]:.3f}")
