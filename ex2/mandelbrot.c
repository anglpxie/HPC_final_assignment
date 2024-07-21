#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
#include <omp.h>

void save_pgm(const short int* image, int width, int height, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    fprintf(file, "P2\n%d %d\n256\n", width, height);

    for (long unsigned int i = 0; i < (long unsigned int) (width * height); i++) {
        fprintf(file, "%d ", image[i]);
        if ((i + 1) % width == 0) {
            fprintf(file, "\n");
        }
    }

    fclose(file);
}

int compute_pixel(int px_x, int px_y, double x_min, double x_max, double y_min, double y_max, int width, int height, int max_iterations) {
    double x = x_min + px_x * (x_max - x_min) / width;
    double y = y_min + px_y * (y_max - y_min) / height;
    double complex c = x + y * I; // I is the imaginary unit defined in complex.h
    double complex z = 0 + 0 * I;

    int iter = 0;
    while (iter < max_iterations && cabs(z) < 2.0) {
        z = z * z + c;
        iter++;
    }
    if (iter == max_iterations) {
        return 0;
    }

    iter = 255 * iter / max_iterations;

    return iter;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        fprintf(stderr, "Usage: %s <X_MIN> <X_MAX> <Y_MIN> <Y_MAX> <width> <height> <MAX_ITER>\n", argv[0]);
        return 1;
    }

    double X_MIN = atof(argv[1]);
    double X_MAX = atof(argv[2]);
    double Y_MIN = atof(argv[3]);
    double Y_MAX = atof(argv[4]);
    int WIDTH = atoi(argv[5]);
    int HEIGHT = atoi(argv[6]);
    int MAX_ITER = atoi(argv[7]);

    int num_pixels = WIDTH * HEIGHT;

    MPI_Init(&argc, &argv);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    double start_time = MPI_Wtime();

    int pixels_per_process = (num_pixels + np - 1) / np; // ceil(num_pixels / np)

    short int* iterations = (short int*) malloc(pixels_per_process * sizeof(short int));
    for (int i = 0; i < pixels_per_process; i++) {
        iterations[i] = -1;
    }

    // each process computes
    #pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < pixels_per_process; i++) {
        int idx = i * np + rank;
        if (idx < num_pixels) {
            iterations[i] = compute_pixel(idx % WIDTH, idx / WIDTH, X_MIN, X_MAX, Y_MIN, Y_MAX, WIDTH, HEIGHT, MAX_ITER);
        }
    }

    if (rank == 0) {
        short int* image = (short int*) malloc(num_pixels * sizeof(short int)); 

        // copy the pixels computed by the master process
        for (int j = 0; j < pixels_per_process; j++) {
            int idx = j * np;
            if (idx < num_pixels && iterations[j] != -1) {
                image[idx] = iterations[j];
            }
        }

        // gather the pixels computed by the worker processes
        for (int i = 1; i < np; i++) {
            MPI_Recv(iterations, pixels_per_process, MPI_SHORT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < pixels_per_process; j++) {
                int idx = i + j * np;
                if (idx < num_pixels && iterations[j] != -1) {
                    image[idx] = iterations[j];
                }
            }
        }
        save_pgm(image, WIDTH, HEIGHT, "output.pgm");
        free(image);
        printf("%f", MPI_Wtime() - start_time);
    } else {
        MPI_Send(iterations, pixels_per_process, MPI_SHORT, 0, 0, MPI_COMM_WORLD);
    }

    free(iterations);
    MPI_Finalize();
    return 0;
}