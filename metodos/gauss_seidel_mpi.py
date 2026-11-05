"""Gauss-Seidel por bloques de filas con mpi4py (RF3).

Cada proceso es dueño de un bloque de filas de A. Dentro de su bloque aplica
Gauss-Seidel (usa valores recién actualizados); para las incógnitas de otros
bloques usa los valores de la iteración anterior. Al final de cada iteración
se intercambian los bloques con Allgatherv.
Con 1 proceso equivale al Gauss-Seidel serial; con más procesos suele
necesitar más iteraciones, pero cada iteración es más barata.

Entradas: A, b, comm, x0, tol (si tol <= 0 se ejecutan exactamente max_iter
iteraciones, útil para medir tiempos), max_iter.
Salida (dict): x, iteraciones, convergio, error_final, historial.
"""
import numpy as np
from mpi4py import MPI


def particion(n, size):
    """Reparte n filas en 'size' bloques contiguos casi iguales."""
    base, resto = divmod(n, size)
    counts = [base + (1 if r < resto else 0) for r in range(size)]
    displs = [sum(counts[:r]) for r in range(size)]
    return counts, displs


def gauss_seidel_mpi(A, b, comm=None, x0=None, tol=1e-10, max_iter=10000):
    comm = comm or MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    A = np.ascontiguousarray(A, dtype=float)
    b = np.ascontiguousarray(b, dtype=float)
    n = len(b)
    if A.shape != (n, n):
        raise ValueError("A debe ser cuadrada y coincidir con el tamaño de b")
    if np.any(np.diag(A) == 0):
        raise ValueError("Hay ceros en la diagonal de A")

    counts, displs = particion(n, size)
    ini, m = displs[rank], counts[rank]
    fin = ini + m
    A_loc = A[ini:fin].copy()
    b_loc = b[ini:fin].copy()
    diag = A_loc[np.arange(m), np.arange(ini, fin)].copy()

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    historial = []
    err = np.inf

    for k in range(1, max_iter + 1):
        xw = x.copy()                       # valores viejos de otros bloques
        for i in range(m):
            g = ini + i
            s = A_loc[i] @ xw - diag[i] * xw[g]
            xw[g] = (b_loc[i] - s) / diag[i]
        x_loc = xw[ini:fin].copy()

        x_nuevo = np.empty(n)
        comm.Allgatherv(x_loc, [x_nuevo, counts, displs, MPI.DOUBLE])

        err = np.linalg.norm(x_nuevo - x, np.inf) / max(np.linalg.norm(x_nuevo, np.inf), 1e-30)
        x = x_nuevo
        historial.append(err)
        if tol > 0 and err < tol:
            return {"x": x, "iteraciones": k, "convergio": True,
                    "error_final": err, "historial": historial}

    return {"x": x, "iteraciones": max_iter, "convergio": False,
            "error_final": err, "historial": historial}
