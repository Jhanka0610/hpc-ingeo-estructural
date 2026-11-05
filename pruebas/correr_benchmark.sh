#!/bin/bash
# Uso: bash pruebas/correr_benchmark.sh [n] [repeticiones] [iteraciones_GS]
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
N=${1:-1200}; REP=${2:-5}; ITER=${3:-300}
rm -f resultados/tiempos.csv
for P in 1 2 4; do
  echo ">>> $P proceso(s)"
  mpirun -n $P python -m pruebas.benchmark_speedup --n $N --rep $REP --iter $ITER
done
python -m pruebas.graficar_speedup
