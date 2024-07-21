#!/bin/bash

#SBATCH --job-name=cvetinovic_weak
#SBATCH --nodes=4
#SBATCH --time=60
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --output=cvetinovic_weak.out
module load openMPI/4.1.5/gnu

X_MIN=-2.0
X_MAX=2.0
Y_MIN=-2.0
Y_MAX=2.0
WIDTH=4096
BASE_HEIGHT=128
MAX_ITERATIONS=1024

output_file="./scalability/MPI_weak.csv"
echo "np,height,time" > $output_file

# MPI
for np in $(seq 2 2 96)
do
    seconds=$(mpirun -np ${np} --map-by socket --bind-to socket -x OMP_NUM_THREADS=1 ./mandelbrot $X_MIN $X_MAX $Y_MIN $Y_MAX $WIDTH $((BASE_HEIGHT * np)) $MAX_ITERATIONS)
    echo "${np},$((BASE_HEIGHT * np)),${seconds}" >> $output_file
done


output_file="./scalability/OpenMP_weak.csv"
echo "n_threads,height,time" > $output_file

# OpenMP
for n_threads in $(seq 2 2 22)
do
    seconds=$(mpirun -np 1 --map-by socket --bind-to socket -x OMP_NUM_THREADS=${n_threads} ./mandelbrot $X_MIN $X_MAX $Y_MIN $Y_MAX $WIDTH $((BASE_HEIGHT * n_threads)) $MAX_ITERATIONS)
    echo "${n_threads},$((BASE_HEIGHT * n_threads)),${seconds}" >> $output_file
done