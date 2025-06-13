#!/usr/bin/env python3
"""
Exercício 1 - Análise de Sinais

Letra a: Geração de malha e função degrau
Letra b: Produto interno por números complexos e integrais reais
Letra c: Coeficientes de Fourier diretos e via FFT
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Letra a: Função degrau unitário
# ----------------------------
def degrau_unitario(v):
    """Função degrau unitário: 0 para v<0, 1 para v>=0."""
    return np.where(v >= 0, 1, 0)

def letra_a():
    print("\n--- Letra a: Função Degrau Unitário ---")
    T = 10
    N = 1000
    inicio = -5
    fim = 5
    passo = T / N

    x = np.arange(inicio, fim, passo)
    y = degrau_unitario(x)

    print(f"Comprimento de x: {x.size}")  # Deve ser 1000
    print(f"Comprimento de y: {y.size}")  # Deve ser 1000

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Função Degrau Unitário")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Letra b.1: Produto interno via números complexos
# ----------------------------
def produto_interno_complexo():
    print("\n--- Letra b.1: Produto Interno via Números Complexos ---")
    N = 1000
    js = np.arange(1, 11)

    inner_products = []
    for j in js:
        n = np.arange(N)
        phi_j = np.exp(1j * 2 * np.pi * j * n / N)
        ip = np.vdot(phi_j, phi_j)
        inner_products.append(ip.real)

    print("j   ⟨φ_j,φ_j⟩")
    print("--- --------")
    for j, ip in zip(js, inner_products):
        print(f"{j:2d}   {ip:.0f}")

# ----------------------------
# Letra b.2: Produto interno via base real (integral)
# ----------------------------
def produto_interno_real():
    print("\n--- Letra b.2: Produto Interno via Integral Real (base cos) ---")
    T = 10
    N = 1000
    dx = T / N
    x = np.linspace(-5, 5, N, endpoint=False)

    norms = np.zeros(10)
    for j in range(1, 11):
        phi = np.cos(2 * np.pi * j * x / T)
        norms[j - 1] = np.sum(phi ** 2) * dx

    print("⟨ϕ_j,ϕ_j⟩ (j = 1..10):", np.round(norms, 6))

# ----------------------------
# Letra c: Coeficientes de Fourier diretos e via FFT
# ----------------------------
def letra_c():
    print("\n--- Letra c: Coeficientes de Fourier Reais e Complexos ---")
    T = 10.0
    N = 1000
    dx = T / N

    x = np.linspace(-5, 5, N, endpoint=False)
    y = np.where(x >= 0, 1.0, 0.0)

    # b) Produto interno base cos
    norms = np.zeros(10)
    for j in range(1, 11):
        phi = np.cos(2 * np.pi * j * x / T)
        norms[j-1] = np.sum(phi**2) * dx
    print("⟨ϕ_j,ϕ_j⟩ (j=1..10):", np.round(norms, 6))

    # c1) Coeficientes reais diretos
    a_direct = np.zeros(11)
    b_direct = np.zeros(11)
    a_direct[0] = (1/T) * np.sum(y) * dx
    for j in range(1, 11):
        a_direct[j] = (2/T) * np.sum(y * np.cos(2*np.pi*j*x/T)) * dx
        b_direct[j] = (2/T) * np.sum(y * np.sin(2*np.pi*j*x/T)) * dx

    print("\nCoeficientes diretos (a_j e b_j):")
    for j in range(11):
        print(f" a[{j:2d}] = {a_direct[j]: .6f},   b[{j:2d}] = {b_direct[j]: .6f}")

    # c2) Coeficientes complexos via FFT
    Y = np.fft.fft(y)
    Yh = Y[:11]  # harmônicos 0 a 10
    c = np.zeros(11, dtype=complex)
    c[0] = Yh[0] / N
    for j in range(1, 11):
        c[j] = Yh[j] / (N / 2)

    print("\nCoeficientes complexos c_j (via FFT):")
    for j in range(11):
        re, im = c[j].real, c[j].imag
        print(f" c[{j:2d}] = {re:+.6f}{im:+.6f}j")

# ----------------------------
# Execução principal
# ----------------------------
if __name__ == "__main__":
    letra_a()
    produto_interno_complexo()
    produto_interno_real()
    letra_c()