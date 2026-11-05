"""Método de Gauss-Jordan con pivoteo parcial (versión serial).

Entradas:
    A   : matriz de coeficientes (n x n)
    b   : vector independiente (n)
    tol : umbral para considerar un pivote como cero (sistema singular)

Salida (dict):
    x (o None), estado ("ok" o "singular")
"""
import numpy as np


def gauss_jordan(A, b, tol=1e-12):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if A.shape != (n, n):
        raise ValueError("A debe ser cuadrada y coincidir con el tamaño de b")

    M = np.hstack([A, b.reshape(-1, 1)])  # matriz aumentada

    for i in range(n):
        p = i + int(np.argmax(np.abs(M[i:, i])))  # pivoteo parcial
        if abs(M[p, i]) < tol:
            return {"x": None, "estado": "singular"}
        if p != i:
            M[[i, p]] = M[[p, i]]
        M[i] = M[i] / M[i, i]
        for j in range(n):
            if j != i:
                M[j] = M[j] - M[j, i] * M[i]

    return {"x": M[:, -1].copy(), "estado": "ok"}
