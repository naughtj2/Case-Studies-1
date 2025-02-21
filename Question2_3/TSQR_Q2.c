#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>

#define M 16  // Total number of rows (must be divisible by 4)
#define N 4   // Number of columns (M > N)
#define NUM_NODES 4  // Number of compute nodes

// Function to perform QR factorization using LAPACK
void qr_factorization(double *A, int rows, int cols, double *Q, double *R) {
    int lda = cols, lwork = cols * 64, info;
    double tau[cols], work[lwork];

    // Copy A to avoid modifying the original matrix
    double A_copy[rows * cols];
    for (int i = 0; i < rows * cols; i++) A_copy[i] = A[i];

    // Compute QR factorization
    info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows, cols, A_copy, cols, tau);
    if (info != 0) {
        printf("Error in QR factorization: %d\n", info);
        exit(1);
    }

    // Extract R matrix
    for (int i = 0; i < cols; i++)
        for (int j = i; j < cols; j++)
            R[i * cols + j] = A_copy[i * cols + j];

    // Compute Q explicitly
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, cols, cols, A_copy, cols, tau);
    if (info != 0) {
        printf("Error in computing Q: %d\n", info);
        exit(1);
    }

    // Copy the computed Q
    for (int i = 0; i < rows * cols; i++) Q[i] = A_copy[i];
}

// Function to print a matrix
void print_matrix(const char *name, double *A, int rows, int cols) {
    printf("%s =\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%8.4f ", A[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

printf("Rank %d: Detected %d processes.\n", rank, size); 

if (size != NUM_NODES) {
    if (rank == 0) printf("Error: This program must be run with %d processes.\n", NUM_NODES);
    MPI_Finalize();
    return 1;
}

    int local_rows = M / NUM_NODES;  // Rows per node
    double A[M * N], local_A[local_rows * N], local_Q[local_rows * N], local_R[N * N];
    double R_stacked[NUM_NODES * N * N], Qr[N * N], R_final[N * N];
    double Q_final[M * N];

    if (rank == 0) {
        // Generate a random matrix
        srand(42);
        for (int i = 0; i < M * N; i++)
            A[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        
        print_matrix("Original Matrix A", A, M, N);
    }

    // Scatter the matrix to all nodes
    MPI_Scatter(A, local_rows * N, MPI_DOUBLE, local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each node performs local QR factorization
    qr_factorization(local_A, local_rows, N, local_Q, local_R);

    // Gather the R matrices from all nodes to root process
    MPI_Gather(local_R, N * N, MPI_DOUBLE, R_stacked, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Perform global QR factorization on stacked R matrices
        qr_factorization(R_stacked, NUM_NODES * N, N, Qr, R_final);

        print_matrix("Final R Matrix", R_final, N, N);
    }

    // Broadcast Qr to all nodes
    MPI_Bcast(Qr, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute final Q = [Q1; Q2; Q3; Q4] * Qr
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            Q_final[(rank * local_rows + i) * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                Q_final[(rank * local_rows + i) * N + j] += local_Q[i * N + k] * Qr[k * N + j];
            }
        }
    }

    // Gather Q_final from all nodes
    MPI_Gather(Q_final + rank * local_rows * N, local_rows * N, MPI_DOUBLE, Q_final, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_matrix("Final Q Matrix", Q_final, M, N);
    }

    MPI_Finalize();
    return 0;
}
