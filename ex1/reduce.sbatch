#!/bin/bash

#SBATCH --job-name=acvetinovic_reduce
#SBATCH --time=120
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --output=reduce.out

module load openMPI/4.1.5/gnu

for np in $(seq 2 2 48)
do
    # We are going to test: linear, chain, binary
    for alg in 1 2 4
    do
        mpirun -np $np --map-by core --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_reduce_algorithm $alg ./osu/osu_reduce --full --iterations 10000 --warmup 1000 --message-size 1:1048576 -f csv >> ./benchmarks/reduce-np$np-a$alg.csv
    done
done
