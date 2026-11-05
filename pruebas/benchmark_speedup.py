"""Mide tiempos de los métodos paralelos y los agrega a resultados/tiempos.csv.
Ejecutar con: mpirun -n P python -m pruebas.benchmark_speedup --n 1200 --rep 5 --iter 300"""
import argparse
import csv
import os

import numpy as np
from mpi4py import MPI
from metodos.gauss_seidel_mpi import gauss_seidel_mpi
from metodos.gauss_jordan_mpi import gauss_jordan_mpi


def matriz_dominante(n, semilla=42):
    rng = np.random.default_rng(semilla)
    A = rng.uniform(-1, 1, (n, n))
    A += np.diag(np.abs(A).sum(axis=1) + 1)
    b = rng.uniform(-10, 10, n)
    return A, b


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--rep", type=int, default=5)
ap.add_argument("--iter", type=int, default=300)
ap.add_argument("--salida", default="resultados/tiempos.csv")
args = ap.parse_args()

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
A, b = matriz_dominante(args.n)

if rank == 0:
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    nuevo = not os.path.exists(args.salida)
    f = open(args.salida, "a", newline="")
    w = csv.writer(f)
    if nuevo:
        w.writerow(["metodo", "n", "procesos", "repeticion", "tiempo_s"])

for metodo in ("gauss_seidel", "gauss_jordan"):
    for rep in range(args.rep + 1):          # rep 0 = calentamiento
        comm.Barrier()
        t0 = MPI.Wtime()
        if metodo == "gauss_seidel":
            gauss_seidel_mpi(A, b, comm, tol=0, max_iter=args.iter)
        else:
            gauss_jordan_mpi(A, b, comm)
        comm.Barrier()
        t = comm.reduce(MPI.Wtime() - t0, op=MPI.MAX, root=0)
        if rank == 0 and rep > 0:
            w.writerow([metodo, args.n, size, rep, f"{t:.6f}"])
            print(f"{metodo:<13} P={size} rep={rep} t={t:.3f} s")

if rank == 0:
    f.close()
