#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#define n1 2000
#define n2 1800
#define n3 1600

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

	MPI_Comm gridComm, columnComm, rowComm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &gridComm);
	int coords[2], subDims[2];
	MPI_Cart_coords(gridComm, rank, 2, coords);
	subDims[0] = 0; subDims[1] = 1;
	MPI_Cart_sub(gridComm, subDims, &rowComm);
	subDims[0] = 1; subDims[1] = 0;
	MPI_Cart_sub(gridComm, subDims, &columnComm);

	double* A =  NULL, * B = NULL, * C = NULL, * subA, * subB, * subC;
	int sub_n = n1 / dims[0];
	int sub_m = n3 / dims[1];
	subA = (double*)malloc(sizeof(double) * sub_n * n2);
	subB = (double*)malloc(sizeof(double) * n2 * sub_m);
	subC = (double*)malloc(sizeof(double) * sub_n * sub_m);

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

	MPI_Datatype SUB_A;
	MPI_Type_contiguous(sub_n * n2, MPI_DOUBLE, &SUB_A);
	MPI_Type_commit(&SUB_A);
	if (coords[1] == 0) {
		MPI_Scatter(A, 1, SUB_A, subA, 1, SUB_A, 0, columnComm);
	}
	MPI_Bcast(subA, 1, SUB_A, 0, rowComm);
	MPI_Type_free(&SUB_A);

	MPI_Datatype SUB_B;
	MPI_Type_vector(n2, sub_m, n3, MPI_DOUBLE, &SUB_B);
	MPI_Datatype SUB_B_RES;
	int mpi_double_size;
	MPI_Type_size(MPI_DOUBLE, &mpi_double_size);
	MPI_Type_create_resized(SUB_B, 0, sub_m * mpi_double_size , &SUB_B_RES);
	MPI_Type_commit(&SUB_B_RES);
	if (coords[0]==0)
		MPI_Scatter(B, 1, SUB_B_RES, subB, sub_m*n2, MPI_DOUBLE, 0, rowComm);
	MPI_Bcast(subB, sub_m*n2, MPI_DOUBLE, 0, columnComm);
	MPI_Type_free(&SUB_B);
	MPI_Type_free(&SUB_B_RES);

	for (int row = 0; row < sub_n; row++) {
		for (int column = 0; column < sub_m; column++) {
			int current_row = row * sub_m;
			subC[current_row + column] = 0;
			for (int i = 0; i < n2; i++) {
				subC[current_row + column] += subA[row * n2 + i] * subB[i * sub_m + column];
			}
		}
	}
	free(subA);
	free(subB);

	MPI_Datatype SUB_C;
	MPI_Type_contiguous(sub_n * sub_m, MPI_DOUBLE, &SUB_C);
	MPI_Type_commit(&SUB_C);

	MPI_Datatype SUB_C_ROWS, SUB_C_STRIDE;
	MPI_Type_contiguous(sub_n * n3, MPI_DOUBLE, &SUB_C_ROWS);
	MPI_Type_commit(&SUB_C_ROWS);
	MPI_Type_vector(sub_n, sub_m, n3, MPI_DOUBLE, &SUB_C_STRIDE);
	MPI_Type_commit(&SUB_C_STRIDE);

	double* subCRows = NULL;
	if (coords[1] == 0) {
		subCRows = (double*)malloc(sizeof(double) * sub_n * n3);
		for (int row = 0; row < sub_n; row++) {
			for (int column = 0; column < sub_m; column++) {
				subCRows[row * n3 + column] = subC[row * sub_m + column];
			}
		}
		for (int i = 1; i < dims[1]; i++) {
			MPI_Recv(subCRows + sub_m * i, 1, SUB_C_STRIDE, i, 1111, rowComm, MPI_STATUS_IGNORE);
		}
	}
	else {
		MPI_Send(subC, 1, SUB_C, 0, 1111, rowComm);
	}

	if (coords[1] == 0) {
		MPI_Gather(subCRows, 1, SUB_C_ROWS, C, 1, SUB_C_ROWS, 0, columnComm);
	}
	MPI_Type_free(&SUB_C_ROWS);
	MPI_Type_free(&SUB_C_STRIDE);
	MPI_Type_free(&SUB_C);
	free(subCRows);
	free(subC);

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
