import numpy as np
import matplotlib.pyplot as plt

"""
Exercício 3 - Obtendo o espectro de um sinal 
"""

#Variáveis do programa
T = 20 #Intervalo
N = 1000 #Número de amostras
x = np.linspace(-10, 10, N, endpoint=False) #Definição da malha espaçada por intervalos de 0.02, com exceção de 10
y = (x >= 0).astype(float) #Definição dos valores unitários

#Cálculo do FFT normalizado, magnitude (considerando módulo) e as frequências
f_fft = np.fft.fft(y) / N #FFT Normalizado
magnitude = np.abs(f_fft) #Magnitude
frequencias = np.fft.fftfreq(N, d=T/N) #Frequências

#Como o FFT é simétrico, podemos desconsiderar metade dos resultados obtidos
half_N = N // 2 #Consideração apenas de metade do intervalo
f_pos = frequencias[:half_N] #Definição de um array com apenas a primeira metade das frequências
m_pos = magnitude[:half_N] #Definição de uma array com apenas a primeira metade das magnitudes

#Plot do gráfico 
plt.figure(figsize=(10, 5))
plt.plot(f_pos, m_pos, '*')
plt.title('Espectro do sinal degrau unitário')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
