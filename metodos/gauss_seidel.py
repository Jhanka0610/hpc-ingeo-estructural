"""Método de Gauss-Seidel (versión serial).

Entradas:
    A        : matriz de coeficientes (n x n)
    b        : vector independiente (n)
    x0       : aproximación inicial (por defecto, ceros)
    tol      : tolerancia sobre el cambio relativo entre iteraciones
    max_iter : máximo de iteraciones

Salida (dict):
    x, iteraciones, convergio (bool), error_final, historial
"""
import numpy as np


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if A.shape != (n, n):
        raise ValueError("A debe ser cuadrada y coincidir con el tamaño de b")
    if np.any(np.diag(A) == 0):
        raise ValueError("Hay ceros en la diagonal de A")

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    historial = []
    err = np.inf

    for k in range(1, max_iter + 1):
        x_ant = x.copy()
        for i in range(n):
            suma = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - suma) / A[i, i]
        err = np.linalg.norm(x - x_ant, np.inf) / max(np.linalg.norm(x, np.inf), 1e-30)
        historial.append(err)
        if err < tol:
            return {"x": x, "iteraciones": k, "convergio": True,
                    "error_final": err, "historial": historial}

    return {"x": x, "iteraciones": max_iter, "convergio": False,
            "error_final": err, "historial": historial}
