#!/bin/bash
#SBATCH --job-name=resolver
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/hpc/scripts/resultados/resolver_%j.out
#SBATCH --error=/mnt/hpc/scripts/resultados/resolver_%j.err

# Uso:  sbatch --export=ALL,ENTRADA=entradas/lineal_n400.json slurm/resolver.slurm
# Opcionales: SALIDA=<carpeta> (por defecto resultados/job_<id>)  MPI_TIPO=pmi2
# Deja en la carpeta de salida: resultado.json y metadatos.json (RF4)
TIPO=${MPI_TIPO:-pmix}
REPO=${REPO:-/mnt/hpc/scripts}

if [ -z "$ENTRADA" ]; then
    echo "ERROR: falta ENTRADA. Uso: sbatch --export=ALL,ENTRADA=ruta.json slurm/resolver.slurm"
    exit 2
fi
SALIDA=${SALIDA:-resultados/job_$SLURM_JOB_ID}

cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$PYTHONPATH
export OMP_NUM_THREADS=1

echo "=== HPC INGEO Estructural: resolver ==="
echo "Trabajo: $SLURM_JOB_ID | Nodos: $SLURM_JOB_NODELIST | Tareas: $SLURM_NTASKS | MPI: $TIPO"
echo "Entrada: $ENTRADA | Salida: $SALIDA"
srun --mpi=$TIPO python3 -m ejecucion.resolver --entrada "$ENTRADA" --salida "$SALIDA"
ESTADO=$?
echo "Estado final: $ESTADO"
exit $ESTADO
