#!/bin/bash
#SBATCH --job-name=test_mpi
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:03:00
#SBATCH --output=/mnt/hpc/scripts/resultados/test_mpi_%j.out
#SBATCH --error=/mnt/hpc/scripts/resultados/test_mpi_%j.err

export OMP_NUM_THREADS=1
PRUEBA='from mpi4py import MPI; c=MPI.COMM_WORLD; print("rank", c.Get_rank(), "de", c.Get_size(), "en", MPI.Get_processor_name())'

echo "Nodos: $SLURM_JOB_NODELIST | tareas: $SLURM_NTASKS"
echo "--- Plugins MPI de Slurm ---"
srun --ntasks=1 --mpi=list 2>&1
echo "--- Prueba con pmix ---"
timeout 60 srun --mpi=pmix python3 -c "$PRUEBA" 2>&1
echo "--- Prueba con pmi2 ---"
timeout 60 srun --mpi=pmi2 python3 -c "$PRUEBA" 2>&1
