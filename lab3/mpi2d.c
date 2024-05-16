#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#define n1 2000
#define n2 1800
#define n3 1600

void scat_and_broadcast_A(int coords[2], int sub_n, double *A, double *sub_A, MPI_Comm column_comm, MPI_Comm row_comm) {
	MPI_Datatype SUB_A;
	MPI_Type_contiguous(sub_n * n2, MPI_DOUBLE, &SUB_A);
	MPI_Type_commit(&SUB_A);
	if (coords[1] == 0) {
		MPI_Scatter(A, 1, SUB_A, sub_A, 1, SUB_A, 0, column_comm);
	}
	MPI_Bcast(sub_A, 1, SUB_A, 0, row_comm);
	MPI_Type_free(&SUB_A);
}

void scat_and_broadcast_B(int coords[2], int sub_m, double *B, double *sub_B, MPI_Comm row_comm, MPI_Comm column_comm) {
	if (coords[0] == 0) {
		MPI_Datatype SUB_B;
		MPI_Type_vector(n2, sub_m, n3, MPI_DOUBLE, &SUB_B);
		MPI_Datatype SUB_B_RES;
		int mpi_double_size;
		MPI_Type_size(MPI_DOUBLE, &mpi_double_size);
		MPI_Type_create_resized(SUB_B, 0, sub_m * mpi_double_size, &SUB_B_RES);
		MPI_Type_commit(&SUB_B_RES);
		MPI_Scatter(B, 1, SUB_B_RES, sub_B, sub_m * n2, MPI_DOUBLE, 0, row_comm);
		MPI_Type_free(&SUB_B);
		MPI_Type_free(&SUB_B_RES);
	}
	MPI_Bcast(sub_B, sub_m * n2, MPI_DOUBLE, 0, column_comm);
}

void print_matrix(const double* matrix, int rows, int columns) {
	for (int x = 0; x < rows; ++x) {
		for (int y = 0; y < columns; ++y) {
			printf("%lf ", matrix[x * columns + y]);
		}
		printf("\n");
	}
	printf("\n");
}

void mult_submatrixs(int sub_n, int sub_m, double *sub_C, double *sub_A, double *sub_B) {
	for (int row = 0; row < sub_n; row++) {
		for (int column = 0; column < sub_m; column++) {
			int current_row = row * sub_m;
			sub_C[current_row + column] = 0;
			for (int i = 0; i < n2; i++) {
				sub_C[current_row + column] += sub_A[row * n2 + i] * sub_B[i * sub_m + column];
			}
		}
	}
}

void collect_sub_C(int sub_n, int sub_m, int coords[2],int dims[2],double *C, double *sub_C,MPI_Comm row_comm, MPI_Comm column_comm) {
	MPI_Datatype SUB_C_ROWS, SUB_C;
	MPI_Type_contiguous(sub_n * n3, MPI_DOUBLE, &SUB_C_ROWS);
	MPI_Type_commit(&SUB_C_ROWS);
	MPI_Type_vector(sub_n, sub_m, n3, MPI_DOUBLE, &SUB_C);
	MPI_Type_commit(&SUB_C);
	double* subCRows = NULL;
	if (coords[1] == 0) {
		subCRows = (double*)malloc(sizeof(double) * sub_n * n3);
		for (int row = 0; row < sub_n; row++) {
			for (int column = 0; column < sub_m; column++) {
				subCRows[row * n3 + column] = sub_C[row * sub_m + column];
			}
		}
		for (int i = 1; i < dims[1]; i++) {
			MPI_Recv(subCRows + sub_m * i, 1, SUB_C, i, 1111, row_comm, MPI_STATUS_IGNORE);
		}
	}
	else {
		MPI_Send(sub_C, sub_n * sub_m, MPI_DOUBLE, 0, 1111, row_comm);
	}

	if (coords[1] == 0) {
		MPI_Gather(subCRows, 1, SUB_C_ROWS, C, 1, SUB_C_ROWS, 0, column_comm);
	}
	MPI_Type_free(&SUB_C_ROWS);
	MPI_Type_free(&SUB_C);
	free(subCRows);
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	double startTime, endTime;
	int dims[2] = { 0, 0 }, periods[2] = { 0, 0 }, reorder = 0;
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (argc == 3) {
		dims[0] = atoi(argv[1]);
		dims[1] = atoi(argv[2]);
	}
	else {
		MPI_Dims_create(size, 2, dims);
	}
	if (rank == 0) printf("DIMS: %d %d\n", dims[0], dims[1]);
	if ((n1 % dims[0] != 0) || (n3 % dims[1] != 0)) {
		if (rank == 0) printf("n1,n3 must be divisible by p1,p2\n");
		return 1;
	}

	MPI_Comm grid_comm, column_comm, row_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
	int coords[2], subDims[2];
	MPI_Cart_coords(grid_comm, rank, 2, coords);
	subDims[0] = 0; subDims[1] = 1;
	MPI_Cart_sub(grid_comm, subDims, &row_comm);
	subDims[0] = 1; subDims[1] = 0;
	MPI_Cart_sub(grid_comm, subDims, &column_comm);

	double* A = NULL, * B = NULL, * C = NULL, * sub_A, * sub_B, * sub_C;
	int sub_n = n1 / dims[0];
	int sub_m = n3 / dims[1];
	sub_A = (double*)malloc(sizeof(double) * sub_n * n2);
	sub_B = (double*)malloc(sizeof(double) * n2 * sub_m);
	sub_C = (double*)malloc(sizeof(double) * sub_n * sub_m);

	if ((coords[0] == 0) && (coords[1] == 0)) {
		A = (double*)malloc(sizeof(double) * n1 * n2);
		B = (double*)malloc(sizeof(double) * n2 * n3);
		C = (double*)malloc(sizeof(double) * n1 * n3);

		for (int i = 0; i < n1 * n2; i++) {
			A[i] = i;
		}
		for (int i = 0; i < n2 * n3; i++) {
			B[i] = i;
		}
		startTime = MPI_Wtime();
	}

	scat_and_broadcast_A(coords, sub_n, A, sub_A, column_comm, row_comm);
	scat_and_broadcast_B(coords, sub_m, B, sub_B, row_comm, column_comm);
	mult_submatrixs(sub_n, sub_m, sub_C, sub_A, sub_B);
	free(sub_A);
	free(sub_B);
	collect_sub_C(sub_n, sub_m, coords, dims, C, sub_C, row_comm, column_comm);
	free(sub_C);
	if (rank == 0) {
		endTime = MPI_Wtime();
		printf("Result: %lf seconds left\n", endTime - startTime);
		free(A);
		free(B);
		free(C);
	}
	MPI_Finalize();
	return 0;
}
