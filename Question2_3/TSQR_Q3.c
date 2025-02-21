#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <time.h>

#define NUM_NODES 4  // Number of compute nodes

// Function to perform QR factorization using LAPACK
void qr_factorization(double *A, int rows, int cols, double *Q, double *R) {
    int lwork = cols * 64, info;
    double tau[cols], work[lwork];

    double A_copy[rows * cols];
    for (int i = 0; i < rows * cols; i++) A_copy[i] = A[i];

    info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows, cols, A_copy, cols, tau);
    if (info != 0) {
        printf("Error in QR factorization: %d\n", info);
        exit(1);
    }

    for (int i = 0; i < cols; i++)
        for (int j = i; j < cols; j++)
            R[i * cols + j] = A_copy[i * cols + j];

    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, cols, cols, A_copy, cols, tau);
    if (info != 0) {
        printf("Error in computing Q: %d\n", info);
        exit(1);
    }

    for (int i = 0; i < rows * cols; i++) Q[i] = A_copy[i];
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_rows> <num_cols>\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    if (M % NUM_NODES != 0 || M <= N) {
        printf("Error: M must be divisible by %d and greater than N.\n", NUM_NODES);
        return 1;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_NODES) {
        if (rank == 0) printf("Error: This program must be run with %d processes.\n", NUM_NODES);
        MPI_Finalize();
        return 1;
    }

    int local_rows = M / NUM_NODES;
    double A[M * N], local_A[local_rows * N], local_Q[local_rows * N], local_R[N * N];
    double R_stacked[NUM_NODES * N * N], Qr[N * N], R_final[N * N];
    double Q_final[M * N];

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < M * N; i++)
            A[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    MPI_Scatter(A, local_rows * N, MPI_DOUBLE, local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    clock_t start = clock();
    qr_factorization(local_A, local_rows, N, local_Q, local_R);
    MPI_Gather(local_R, N * N, MPI_DOUBLE, R_stacked, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        qr_factorization(R_stacked, NUM_NODES * N, N, Qr, R_final);
    }

    MPI_Bcast(Qr, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            Q_final[(rank * local_rows + i) * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                Q_final[(rank * local_rows + i) * N + j] += local_Q[i * N + k] * Qr[k * N + j];
            }
        }
    }

    MPI_Gather(Q_final + rank * local_rows * N, local_rows * N, MPI_DOUBLE, Q_final, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    clock_t end = clock();

    if (rank == 0) {
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("M = %d, N = %d, Execution Time: %f seconds\n", M, N, time_taken);
    }

    MPI_Finalize();
    return 0;
}
