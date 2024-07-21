#!/bin/bash

#SBATCH --job-name=cvetinovic_strong
#SBATCH --nodes=4
#SBATCH --time=60
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --output=cvetinovic_strong.out
module load openMPI/4.1.5/gnu

X_MIN=-2.0
X_MAX=2.0
Y_MIN=-2.0
Y_MAX=2.0
WIDTH=4096
HEIGHT=4096
MAX_ITERATIONS=1024

output_file="./scalability/MPI_strong_scalability.csv"
echo "np,time" > $output_file

# MPI
for np in $(seq 2 2 96)
do
    output=$(mpirun -np ${np} --map-by socket --bind-to socket -x OMP_NUM_THREADS=1 ./mandelbrot $X_MIN $X_MAX $Y_MIN $Y_MAX $WIDTH $HEIGHT $MAX_ITERATIONS)
    echo "${np},${output}" >> $output_file
done


output_file="./scalability/OMP_strong_scalability.csv"
echo "n_threads,time" > $output_file

# OpenMP
for n_threads in $(seq 2 2 22)
do
    output=$(mpirun -np 1 --map-by socket --bind-to none -x OMP_NUM_THREADS=${n_threads} ./mandelbrot $X_MIN $X_MAX $Y_MIN $Y_MAX $WIDTH $HEIGHT $MAX_ITERATIONS)
    echo "${n_threads},${output}" >> $output_file
done