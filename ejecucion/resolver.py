"""Resuelve un sistema lineal leido de un archivo JSON y guarda resultado + metadatos (RF4).

Uso (serie):      python3 -m ejecucion.resolver --entrada problema.json --salida resultados/mi_corrida
Uso (paralelo):   mpirun -n 4 python3 -m ejecucion.resolver --entrada problema.json --salida resultados/mi_corrida
En Slurm:         srun --mpi=pmix python3 -m ejecucion.resolver --entrada ... --salida ...

Formato del JSON de entrada:
  {"metodo": "gauss_seidel" | "gauss_jordan",
   "A": [[...], ...], "b": [...],
   "tol": 1e-10, "max_iter": 10000}       (tol y max_iter son opcionales y solo aplican a gauss_seidel)

Genera en la carpeta de salida: resultado.json y metadatos.json
"""
import argparse
import datetime
import getpass
import json
import os
import sys
import time

import numpy as np

METODOS = ("gauss_seidel", "gauss_jordan")


def cargar_problema(ruta):
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    metodo = datos.get("metodo")
    if metodo not in METODOS:
        raise ValueError(f"'metodo' debe ser uno de {METODOS}, llego: {metodo!r}")
    for clave in ("A", "b"):
        if clave not in datos:
            raise ValueError(f"Falta la clave '{clave}' en el archivo de entrada")
    A = np.array(datos["A"], dtype=float)
    b = np.array(datos["b"], dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A debe ser cuadrada, llego con forma {A.shape}")
    if b.shape != (A.shape[0],):
        raise ValueError(f"b debe tener {A.shape[0]} elementos, llego con forma {b.shape}")
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("A y b no pueden contener NaN ni infinitos")
    tol = float(datos.get("tol", 1e-10))
    max_iter = int(datos.get("max_iter", 10000))
    return metodo, A, b, tol, max_iter


def obtener_comm():
    """Devuelve (comm, rank, size). Sin mpi4py trabaja en serie."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except ImportError:
        return None, 0, 1


def resolver(metodo, A, b, tol, max_iter, comm, size):
    if size > 1:
        if metodo == "gauss_seidel":
            from metodos.gauss_seidel_mpi import gauss_seidel_mpi
            return gauss_seidel_mpi(A, b, comm, tol=tol, max_iter=max_iter)
        from metodos.gauss_jordan_mpi import gauss_jordan_mpi
        return gauss_jordan_mpi(A, b, comm)
    if metodo == "gauss_seidel":
        from metodos.gauss_seidel import gauss_seidel
        return gauss_seidel(A, b, tol=tol, max_iter=max_iter)
    from metodos.gauss_jordan import gauss_jordan
    return gauss_jordan(A, b)


def escribir_json(ruta, objeto):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(objeto, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="archivo JSON con el problema")
    ap.add_argument("--salida", required=True, help="carpeta donde guardar resultado.json y metadatos.json")
    args = ap.parse_args()

    comm, rank, size = obtener_comm()
    inicio = time.perf_counter()
    meta = {
        "usuario": getpass.getuser(),
        "fecha_hora": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "archivo_entrada": os.path.abspath(args.entrada),
        "procesos": size,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "nodos": os.environ.get("SLURM_JOB_NODELIST"),
        "advertencias": [],
    }
    resultado, codigo = {}, 0
    try:
        metodo, A, b, tol, max_iter = cargar_problema(args.entrada)
        meta["metodo"] = metodo
        meta["parametros"] = {"n": int(A.shape[0]), "tol": tol, "max_iter": max_iter}

        esperadas = int(os.environ.get("SLURM_NTASKS", "1"))
        if esperadas > 1 and size == 1:
            meta["advertencias"].append(
                f"Slurm asigno {esperadas} tareas pero MPI solo ve 1 proceso: "
                "revisar --mpi=pmix/pmi2 en srun. Se resolvio en serie.")

        r = resolver(metodo, A, b, tol, max_iter, comm, size)
        if rank == 0:
            x = r.get("x")
            if x is None:
                meta["estado_final"] = r.get("estado", "sin_solucion")
                codigo = 1
            else:
                x = np.asarray(x, dtype=float)
                resid = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
                convergio = r.get("convergio", r.get("estado") == "ok")
                meta["estado_final"] = "ok" if convergio else "no_convergio"
                resultado = {"x": x.tolist(), "iteraciones": r.get("iteraciones"),
                             "residuo_relativo": resid}
                codigo = 0 if convergio else 1
    except Exception as e:  # entrada invalida u otro error: se registra en los metadatos
        meta["estado_final"] = "error"
        meta["detalle_error"] = f"{type(e).__name__}: {e}"
        codigo = 2

    if rank == 0:
        meta["tiempo_s"] = round(time.perf_counter() - inicio, 6)
        os.makedirs(args.salida, exist_ok=True)
        escribir_json(os.path.join(args.salida, "metadatos.json"), meta)
        if resultado:
            escribir_json(os.path.join(args.salida, "resultado.json"), resultado)
        print(f"[{meta['estado_final']}] metodo={meta.get('metodo')} procesos={size} "
              f"tiempo={meta['tiempo_s']}s -> {args.salida}")
        for a in meta["advertencias"]:
            print("ADVERTENCIA:", a)
    sys.exit(codigo)


if __name__ == "__main__":
    main()
