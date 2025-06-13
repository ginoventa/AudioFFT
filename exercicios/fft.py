import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 10
N = 1000
x = np.arange(-5, 5, T / N)
y = (x >= 0).astype(float)

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

f_fft = np.fft.fft(y)
freqs = np.fft.fftfreq(N, d=T/N)

print("\nFFT (normalizada, primeiros 11 coeficientes):")
for i in range(11):
    print(f"f_fft[{i}] = {f_fft[i]/N:.5f}")

a_fft = [f_fft[0].real / N]

for k in range(1, 6):
    a_sin = 2 * f_fft[k].imag / N
    a_cos = 2 * f_fft[k].real / N
    a_fft.append(a_sin)
    a_fft.append(a_cos)

print("\nCoeficientes a_j extraídos da FFT (normalizados):")
for i, val in enumerate(a_fft):
    print(f"a{i} = {val:.5f}")

print("\nDiferença entre coeficientes (integração - FFT normalizada):")
for i in range(len(coeficientes)):
    diff = coeficientes[i] - a_fft[i]
    print(f"a{i}: diferença = {diff:.5e}")

plt.figure(figsize=(10, 4))
plt.stem(freqs[:N//2], np.abs(f_fft[:N//2]), basefmt=" ", use_line_collection=True)
plt.title("Magnitude da FFT de y")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

print("""
Relação entre coeficientes:
- a0 corresponde ao termo DC f_fft[0].
- Para j>=1: a_{2k-1} (seno) = 2 * imag(f_fft[k]), a_{2k} (cosseno) = 2 * real(f_fft[k]).
- Os coeficientes calculados via integração e via FFT coincidem praticamente (diferenças numéricas pequenas).
""")
