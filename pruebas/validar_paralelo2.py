"""Valida las versiones MPI contra numpy (RT1: error <= 1e-6) con un problema leido de JSON.

Uso:      mpirun -n P python3 -m pruebas.validar_paralelo2 --entrada entradas/ejemplo_lineal.json
En Slurm: srun --mpi=pmix python3 -m pruebas.validar_paralelo2 --entrada entradas/ejemplo_lineal.json
Codigo de salida (proceso 0): 0 si todo OK, 1 si algun metodo FALLA, 2 si la entrada no sirve.
"""
import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

from ejecucion.resolver import cargar_problema
from metodos.gauss_seidel import gauss_seidel
from metodos.gauss_seidel_mpi import gauss_seidel_mpi
from metodos.gauss_jordan_mpi import gauss_jordan_mpi

LIMITE = 1e-6


def error_rel(x, ref):
    return np.linalg.norm(x - ref) / np.linalg.norm(ref)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--entrada", required=True, help="archivo JSON con A, b (y opcional tol, max_iter)")
args = ap.parse_args()

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

try:  # todos los procesos leen el mismo archivo, asi que todos fallan o ninguno
    _, A, b, tol, max_iter = cargar_problema(args.entrada)
    ref = np.linalg.solve(A, b)
except Exception as e:
    if rank == 0:
        print(f"ERROR de entrada: {type(e).__name__}: {e}")
    sys.exit(2 if rank == 0 else 0)

n = A.shape[0]
esperadas = int(os.environ.get("SLURM_NTASKS", "1"))
if rank == 0 and esperadas > 1 and size == 1:
    print(f"ADVERTENCIA: Slurm asigno {esperadas} tareas pero MPI solo ve 1 proceso (revisar --mpi=pmix/pmi2)")

r_gs = gauss_seidel_mpi(A, b, comm, tol=tol, max_iter=max_iter)
r_gj = gauss_jordan_mpi(A, b, comm)

fallas = 0
if rank == 0:
    serial = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
    print(f"Problema n={n} | {args.entrada} | procesos={size} | Gauss-Seidel serial: {serial['iteraciones']} iter")

    if r_gs["x"] is None:
        e, ok = float("nan"), False
    else:
        e = error_rel(r_gs["x"], ref)
        ok = bool(r_gs["convergio"] and e <= LIMITE)
    fallas += not ok
    print(f"  Gauss-Seidel por bloques  iter={r_gs['iteraciones']:>5}  error={e:.3e}  {'OK' if ok else 'FALLA'}")

    if r_gj["x"] is None:
        e, ok = float("nan"), False
    else:
        e = error_rel(r_gj["x"], ref)
        ok = bool(r_gj["estado"] == "ok" and e <= LIMITE)
    fallas += not ok
    print(f"  Gauss-Jordan por filas    iter={'-':>5}  error={e:.3e}  {'OK' if ok else 'FALLA'}")
    sys.exit(1 if fallas else 0)
