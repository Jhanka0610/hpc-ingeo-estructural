"""Valida Gauss-Seidel y Gauss-Jordan contra numpy.linalg.solve (RT1: error <= 1e-6)."""
import numpy as np
from metodos.gauss_seidel import gauss_seidel
from metodos.gauss_jordan import gauss_jordan

LIMITE = 1e-6


def error_rel(x, ref):
    return np.linalg.norm(x - ref) / np.linalg.norm(ref)


def caso_tridiagonal(n):
    """Matriz tipo rigidez (2 en la diagonal, -1 a los lados): simétrica definida positiva."""
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.ones(n)
    return A, b


def caso_dominante(n, semilla=42):
    rng = np.random.default_rng(semilla)
    A = rng.uniform(-1, 1, (n, n))
    A += np.diag(np.abs(A).sum(axis=1) + 1)  # diagonal dominante
    b = rng.uniform(-10, 10, n)
    return A, b


casos = [("Tridiagonal n=30", *caso_tridiagonal(30)),
         ("Dominante n=100", *caso_dominante(100))]

print(f"{'Caso':<18}{'Método':<14}{'Iter':>7}{'Error relativo':>18}  Resultado")
for nombre, A, b in casos:
    ref = np.linalg.solve(A, b)

    r = gauss_seidel(A, b, tol=1e-12, max_iter=20000)
    e = error_rel(r["x"], ref)
    ok = r["convergio"] and e <= LIMITE
    print(f"{nombre:<18}{'Gauss-Seidel':<14}{r['iteraciones']:>7}{e:>18.3e}  {'OK' if ok else 'FALLA'}")

    r = gauss_jordan(A, b)
    e = error_rel(r["x"], ref)
    ok = r["estado"] == "ok" and e <= LIMITE
    print(f"{nombre:<18}{'Gauss-Jordan':<14}{'-':>7}{e:>18.3e}  {'OK' if ok else 'FALLA'}")

# Caso singular: Gauss-Jordan debe avisar, no fallar
sing = gauss_jordan([[1, 2], [2, 4]], [3, 6])
print("\nSistema singular ->", sing["estado"])
