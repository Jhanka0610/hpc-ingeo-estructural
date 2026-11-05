"""Gauss-Jordan con pivoteo parcial, distribuido por filas con mpi4py (RF3).

En cada paso i: (1) cada proceso propone su mejor candidato a pivote,
(2) se elige el global, (3) se hace bcast de la fila pivote (y de la fila i
si hay intercambio), (4) cada proceso elimina la columna i en sus filas.

Entradas: A, b, comm, tol (umbral de pivote nulo).
Salida (dict): x (o None), estado ("ok" o "singular").
"""
from bisect import bisect_right

import numpy as np
from mpi4py import MPI

from metodos.gauss_seidel_mpi import particion


def gauss_jordan_mpi(A, b, comm=None, tol=1e-12):
    comm = comm or MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if A.shape != (n, n):
        raise ValueError("A debe ser cuadrada y coincidir con el tamaño de b")

    counts, displs = particion(n, size)
    ini, m = displs[rank], counts[rank]
    fin = ini + m
    M = np.hstack([A[ini:fin], b[ini:fin].reshape(-1, 1)])  # bloque local aumentado

    def dueno(g):
        return bisect_right(displs, g) - 1

    for i in range(n):
        # 1) candidato local a pivote (solo filas con índice global >= i)
        lo = max(i, ini)
        if lo < fin:
            col = np.abs(M[lo - ini:, i])
            k = int(np.argmax(col))
            cand = (float(col[k]), lo + k)
        else:
            cand = (-1.0, -1)

        # 2) pivote global (mayor valor; ante empate, menor índice)
        val, p = max(comm.allgather(cand), key=lambda t: (t[0], -t[1]))
        if val < tol:
            return {"x": None, "estado": "singular"}

        d_p, d_i = dueno(p), dueno(i)

        # 3) difundir la fila pivote y, si hay intercambio, la fila i
        fila_p = comm.bcast(M[p - ini].copy() if rank == d_p else None, root=d_p)
        if p != i:
            fila_i = comm.bcast(M[i - ini].copy() if rank == d_i else None, root=d_i)
            if rank == d_p:
                M[p - ini] = fila_i
            if rank == d_i:
                M[i - ini] = fila_p
        fila_p = fila_p / fila_p[i]          # fila pivote normalizada

        # 4) eliminación en las filas locales
        factores = M[:, i].copy()
        if rank == d_i:
            factores[i - ini] = 0.0
        M -= np.outer(factores, fila_p)
        if rank == d_i:
            M[i - ini] = fila_p

    x = np.empty(n)
    comm.Allgatherv(M[:, -1].copy(), [x, counts, displs, MPI.DOUBLE])
    return {"x": x, "estado": "ok"}
