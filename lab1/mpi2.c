#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000
#define EPSILON 1E-7
#define TAU 1E-5

void set_matrix_part(int* line_counts, int* line_offsets, int size, int process_count);
void generate_A_chunk(double* A_chunk, int line_count, int line_size, int process_rank);
void generate_x_chunk(double* x_chunk, int size);
void generate_b_chunk(double* b_chunk, int size);
double calc_norm_square(double* vector, int size);
void calc_Axb(const double* A_chunk, const double* x_chunk, const double* b_chunk, double* replace_x_chunk,
    double* Axb_chunk, int* line_counts, int* line_offsets, int process_rank, int process_count);
void calc_next_x(const double* Axb, double* x_chunk, double tau, int chunk_size);
void copy_vector(double* dest, const double* src, int size);

int main(int argc, char** argv)
{
    int process_rank;
    int process_count;
    int iter_count;
    double b_chunk_norm;
    double b_norm;
    double x_chunk_norm;
    double x_norm;
    double Axb_chunk_norm_square;
    double accuracy = EPSILON + 1;
    double start_time;
    double finish_time;
    int* line_counts;
    int* line_offsets;
    double* x_chunk;
    double* b_chunk;
    double* A_chunk;
    double* Axb_chunk;
    double* replace_x_chunk;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    line_counts = malloc(sizeof(int) * process_count);
    line_offsets = malloc(sizeof(int) * process_count);
    set_matrix_part(line_counts, line_offsets, N, process_count);

    x_chunk = malloc(sizeof(double) * line_counts[process_rank]);
    b_chunk = malloc(sizeof(double) * line_counts[process_rank]);
    A_chunk = malloc(sizeof(double) * line_counts[process_rank] * N);
    generate_x_chunk(x_chunk, line_counts[process_rank]);
    generate_b_chunk(b_chunk, line_counts[process_rank]);
    generate_A_chunk(A_chunk, line_counts[process_rank], N, line_offsets[process_rank]);

    b_chunk_norm = calc_norm_square(b_chunk, line_counts[process_rank]);
    MPI_Reduce(&b_chunk_norm, &b_norm, 1, MPI_DOUBLE,
        MPI_SUM, 0, MPI_COMM_WORLD);
    if (process_rank == 0)
        b_norm = sqrt(b_norm);

    Axb_chunk = malloc(sizeof(double) * line_counts[process_rank]);
    replace_x_chunk = malloc(sizeof(double) * line_counts[0]);

    start_time = MPI_Wtime();

    for (iter_count = 0; accuracy > EPSILON; ++iter_count){
        calc_Axb(A_chunk, x_chunk, b_chunk, replace_x_chunk, Axb_chunk,
            line_counts, line_offsets, process_rank, process_count);

        calc_next_x(Axb_chunk, x_chunk, TAU, line_counts[process_rank]);

        Axb_chunk_norm_square = calc_norm_square(Axb_chunk, line_counts[process_rank]);
        MPI_Reduce(&Axb_chunk_norm_square, &accuracy, 1,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (process_rank == 0)
            accuracy = sqrt(accuracy) / b_norm;
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    finish_time = MPI_Wtime();
    x_chunk_norm = calc_norm_square(x_chunk, line_counts[process_rank]);
    MPI_Reduce(&x_chunk_norm, &x_norm, 1, MPI_DOUBLE,
        MPI_SUM, 0, MPI_COMM_WORLD);
    if (process_rank == 0)
    {
            printf("Norm: %lf\n", sqrt(x_norm));
            printf("Time: %lf sec\n", finish_time - start_time);
    }
    free(line_counts);
    free(line_offsets);
    free(x_chunk);
    free(b_chunk);
    free(A_chunk);
    free(Axb_chunk);
    MPI_Finalize();
    return 0;
}

void generate_A_chunk(double* A_chunk, int line_count, int line_size, int lineIndex){
    for (int i = 0; i < line_count; i++){
        for (int j = 0; j < line_size; ++j)
            A_chunk[i * line_size + j] = 1;
        A_chunk[i * line_size + lineIndex + i] = 2;
    }
}

void generate_x_chunk(double* x_chunk, int size){
    for (int i = 0; i < size; i++)
        x_chunk[i] = 0;
}

void generate_b_chunk(double* b_chunk, int size){
    for (int i = 0; i < size; i++)
        b_chunk[i] = N + 1;
}

void set_matrix_part(int* line_counts, int* line_offsets, int size, int process_count){
    int offset = 0;
    for (int i = 0; i < process_count; ++i){
        line_counts[i] = size / process_count;
        if (i < size % process_count)
            ++line_counts[i];
        line_offsets[i] = offset;
        offset += line_counts[i];
    }
}

double calc_norm_square(double* vector, int size){
    double norm_square = 0.0;
    for (int i = 0; i < size; ++i)
        norm_square += vector[i] * vector[i];
    return norm_square;
}

void calc_Axb(const double* A_chunk, const double* x_chunk, const double* b_chunk, double* replace_x_chunk,
    double* Axb_chunk, int* line_counts, int* line_offsets, int process_rank, int process_count){
    int src_rank = (process_rank + process_count - 1) % process_count;
    int dest_rank = (process_rank + 1) % process_count;
    int current_rank;
    copy_vector(replace_x_chunk, x_chunk, line_counts[process_rank]);
    for (int i = 0; i < process_count; ++i)  {
        current_rank = (process_rank + i) % process_count;
        for (int j = 0; j < line_counts[process_rank]; ++j){
            if (i == 0)
                Axb_chunk[j] = -b_chunk[j];
            for (int k = 0; k < line_counts[current_rank]; ++k)
                Axb_chunk[j] += A_chunk[j * N + line_offsets[current_rank] + k] * replace_x_chunk[k];
        }

        if (i != process_count - 1)
            MPI_Sendrecv_replace(replace_x_chunk, line_counts[0], MPI_DOUBLE, dest_rank, process_rank,
                src_rank, src_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void calc_next_x(const double* Axb_chunk, double* x_chunk, double tau, int chunk_size){
    for (int i = 0; i < chunk_size; ++i)
        x_chunk[i] -= tau * Axb_chunk[i];
}

void copy_vector(double* dest, const double* src, int size){
    for (int i = 0; i < size; i++)
        dest[i] = src[i];
}