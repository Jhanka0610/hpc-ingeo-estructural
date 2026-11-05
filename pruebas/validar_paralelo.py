"""Valida las versiones MPI contra numpy (RT1: error <= 1e-6).
Ejecutar con: mpirun -n P python -m pruebas.validar_paralelo"""
import numpy as np
from mpi4py import MPI
from metodos.gauss_seidel import gauss_seidel
from metodos.gauss_seidel_mpi import gauss_seidel_mpi
from metodos.gauss_jordan_mpi import gauss_jordan_mpi

LIMITE = 1e-6
N = 400


def error_rel(x, ref):
    return np.linalg.norm(x - ref) / np.linalg.norm(ref)


comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Matriz tipo rigidez con apoyo elástico (diagonal 2.2): simétrica y dominante
A = 2.2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
b = np.ones(N)
ref = np.linalg.solve(A, b)

r_gs = gauss_seidel_mpi(A, b, comm, tol=1e-12, max_iter=20000)
r_gj = gauss_jordan_mpi(A, b, comm)

if rank == 0:
    serial = gauss_seidel(A, b, tol=1e-12, max_iter=20000)
    print(f"Matriz tridiagonal n={N} | procesos={size} | Gauss-Seidel serial: {serial['iteraciones']} iter")
    e = error_rel(r_gs["x"], ref)
    ok = r_gs["convergio"] and e <= LIMITE
    print(f"  Gauss-Seidel por bloques  iter={r_gs['iteraciones']:>5}  error={e:.3e}  {'OK' if ok else 'FALLA'}")
    e = error_rel(r_gj["x"], ref)
    ok = r_gj["estado"] == "ok" and e <= LIMITE
    print(f"  Gauss-Jordan por filas    iter={'-':>5}  error={e:.3e}  {'OK' if ok else 'FALLA'}")
