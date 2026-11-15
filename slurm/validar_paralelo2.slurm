#!/bin/bash
#SBATCH --job-name=validar2
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/hpc/scripts/resultados/validar2_%j.out
#SBATCH --error=/mnt/hpc/scripts/resultados/validar2_%j.err

# Uso:              sbatch slurm/validar_paralelo2.slurm
# Otro problema:    sbatch --export=ALL,ENTRADA=entradas/otro.json slurm/validar_paralelo2.slurm
# Si pmix no sirve: sbatch --export=ALL,MPI_TIPO=pmi2 slurm/validar_paralelo2.slurm
TIPO=${MPI_TIPO:-pmix}
ENTRADA=${ENTRADA:-entradas/lineal_n400.json}
REPO=${REPO:-/mnt/hpc/scripts}

cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$PYTHONPATH
export OMP_NUM_THREADS=1

echo "=== HPC INGEO Estructural: validacion paralela con datos reales ==="
echo "Trabajo: $SLURM_JOB_ID | Nodos: $SLURM_JOB_NODELIST | Tareas: $SLURM_NTASKS | MPI: $TIPO"
echo "Entrada: $ENTRADA"
srun --mpi=$TIPO python3 -m pruebas.validar_paralelo2 --entrada "$ENTRADA"
ESTADO=$?
echo "Estado final: $ESTADO"
exit $ESTADO
