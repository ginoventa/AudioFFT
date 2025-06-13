import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

##
#Exercício 1 - Relação entre mínimos quadrados trigonométricos e FFT
##

#Variáveis globais do programa
T = 10 #Intervalo
N = 1000 #Número de amostras 
x = np.arange(-5, 5, T / N) #Definição da malha espaçada por intervalos de 0.01, com exceção de 5

#a) Gere a malha x = -5:(10/1000):5, dispense o último ponto, calcule os valores de f nesses pontos e armazene num vetor y.
y = (x >= 0).astype(float) #Definição dos valores unitários

#b) Calcule ⟨ϕj, ϕj ⟩ para j = 1, 2, 3, ..., 10 no computador, e baseado nos resultados descubra a relação com o número de amostras.
produtosInternos = []
for j in range(1, 11):
    k = np.floor((j + 1) / 2)
    if j % 2 == 1:
        integrando = lambda x: np.sin(2 * np.pi * k * x / T) ** 2
    else:
        integrando = lambda x: np.cos(2 * np.pi * k * x / T) ** 2
    val, _ = quad(integrando, -5, 5)
    produtosInternos.append(val)

print("Produtos internos <phi_j, phi_j> para j=1..10:")
for j, val in enumerate(produtosInternos, start=1):
    print(f"<phi_{j}, phi_{j}> ≈ {val:.5f}")

#c) Calcule a0, a1, . . . , a10. No Octave/Matlab, pe¸ca f = fft(y), descarte a última 
# metade do vetor f, e compare os coeficientes com os a obtidos. Qual a relação entre eles?
#c-1)Cálculo dos coeficientes via integral
a0, _ = quad(lambda x: float(x >= 0), -5, 5)
a0 = a0 / 10
coeficientes = [a0]
for j in range(1, 11):
    k = ((j + 1) / 2)
    if j % 2 == 1:
        phi_j = lambda x: np.sin(2 * np.pi * k * x / T)
    else:
        phi_j = lambda x: np.cos(2 * np.pi * k * x / T)
    numerador, _ = quad(lambda x: float(x >= 0) * phi_j(x), -5, 5)
    denominador, _ = quad(lambda x: phi_j(x) ** 2, -5, 5)
    aj = numerador / denominador
    coeficientes.append(aj)
print("\nCoeficientes a0..a10 calculados via integrais:")
for i, c in enumerate(coeficientes):
    print(f"a{i} = {c:.5f}")

#c-2)Cálculo dos coeficientes via FFT
f_fft = np.fft.fft(y)
freqs = np.fft.fftfreq(N, d=T/N)
print("\nFFT (normalizada, primeiros 11 coeficientes):")
for i in range(11):
    print(f"f_fft[{i}] = {f_fft[i]/N:.5f}")

a_fft = [f_fft[0].real / N]

for k in range(1, 6): #Normalização dos coeficientes e extração da parte real e imaginária 
    a_sin = 2 * f_fft[k].imag / N
    a_cos = 2 * f_fft[k].real / N
    a_fft.append(a_sin)
    a_fft.append(a_cos)

print("\nCoeficientes a_j extraídos da FFT (normalizados):")
for i, val in enumerate(a_fft):
    print(f"a{i} = {val:.5f}")
