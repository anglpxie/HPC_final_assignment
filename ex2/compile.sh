module load openMPI/4.1.5/gnu
mpicc -o mandelbrot mandelbrot.c -Wall -Wextra -O3 -march=native -fopenmp -lm
