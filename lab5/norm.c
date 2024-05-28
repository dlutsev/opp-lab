#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <mpi.h>
#include <pthread.h>
#include <stdbool.h>

/* TASK QUEUE */

#define SUCCESS 0
#define ERROR   (-1)

typedef struct {
    int id;
    int process_id;
    int weight;
} Task;

typedef struct {
    Task* task_list;
    int capacity;
    int size;
    int pop_index;
} Task_Queue;

Task_Queue* task_queue_create(int capacity) {
    Task_Queue* task_queue = malloc(sizeof(Task_Queue));

    if (task_queue == NULL) {
        return NULL;
    }

    Task* task_list = malloc(sizeof(Task) * capacity);
    if (task_list == NULL) {
        return NULL;
    }

    task_queue->task_list = task_list;
    task_queue->capacity = capacity;
    task_queue->size = 0;
    task_queue->pop_index = 0;

    return task_queue;
}

bool task_queue_is_empty(const Task_Queue* task_queue) {
    return task_queue->size == 0;
}

bool task_queue_is_full(const Task_Queue* task_queue) {
    return task_queue->size == task_queue->capacity;
}

int task_queue_push(Task_Queue* task_queue, Task task) {
    if (task_queue == NULL) {
        return ERROR;
    }

    if (task_queue_is_full(task_queue)) {
        return ERROR;
    }

    int push_index = (task_queue->pop_index + task_queue->size) % task_queue->capacity;
    task_queue->task_list[push_index] = task;
    task_queue->size++;

    return SUCCESS;
}

int task_queue_pop(Task_Queue* task_queue, Task* task) {
    if (task_queue == NULL) {
        return ERROR;
    }

    if (task_queue_is_empty(task_queue)) {
        return ERROR;
    }

    *task = task_queue->task_list[task_queue->pop_index];
    task_queue->pop_index = (task_queue->pop_index + 1) % task_queue->capacity;
    task_queue->size--;

    return SUCCESS;
}

void task_queue_destroy(Task_Queue** task_queue) {
    if (*task_queue == NULL) {
        return;
    }

    if ((*task_queue)->task_list == NULL) {
        return;
    }

    free((*task_queue)->task_list);
    free(*task_queue);

    *task_queue = NULL;
}

/* TASK QUEUE */

#define TASK_COUNT              10000
#define TOTAL_SUM_WEIGHT        47000000
#define REQUEST_TAG             0
#define RESPONSE_TAG            1
#define EMPTY_QUEUE_RESPONSE    (-1)
#define TERMINATION_SIGNAL      (-2)

static int process_id, process_count;
static int process_start_sum_weight = 0;
static int process_complete_tasks_weight_sum = 0;
bool termination = false;
Task_Queue* task_queue;

pthread_mutex_t mutex;
pthread_cond_t worker_cond;
pthread_cond_t receiver_cond;

static double global_res = 0;

static inline void init_tasks() {
    int min_weight = 2 * TOTAL_SUM_WEIGHT / (TASK_COUNT * (process_count + 1));
    int task_id = 1;

    for (int i = 0; i < TASK_COUNT; ++i) {
        Task task = {
            .id = task_id,
            .process_id = process_id,
            .weight = min_weight * (i % process_count + 1)
        };

        if (i % process_count == process_id) {
            task_queue_push(task_queue, task);
            task_id++;
            process_start_sum_weight += task.weight;
        }
    }
}

static inline void execute_tasks() {
    while (true) {
        Task task;

        pthread_mutex_lock(&mutex);
        if (task_queue_is_empty(task_queue)) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        task_queue_pop(task_queue, &task);
        pthread_mutex_unlock(&mutex);

        for (int i = 0; i < task.weight; ++i) {
            for (int j = 0; j < 180; ++j) {
                global_res += sqrt(sqrt(sqrt(sqrt(i))));
            }
        }

        process_complete_tasks_weight_sum += task.weight;
    }
}

void* worker_start() {
    init_tasks();

    MPI_Barrier(MPI_COMM_WORLD);

    while (true) {
        execute_tasks();
        pthread_mutex_lock(&mutex);
        while (task_queue_is_empty(task_queue) && !termination) {
            pthread_cond_signal(&receiver_cond);
            pthread_cond_wait(&worker_cond, &mutex);
        }

        if (termination) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);
    }

    printf("Worker %d finished\n", process_id);
    pthread_exit(NULL);
}

void* receiver_start() {
    int termination_signal = TERMINATION_SIGNAL;

    while (!termination) {
        int received_tasks = 0;
        Task task;

        pthread_mutex_lock(&mutex);
        while (!task_queue_is_empty(task_queue)) {
            pthread_cond_wait(&receiver_cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        for (int i = 0; i < process_count; ++i) {
            if (i == process_id) {
                continue;
            }

            MPI_Send(&process_id, 1, MPI_INT, i, REQUEST_TAG, MPI_COMM_WORLD);
            MPI_Recv(&task, sizeof(task), MPI_BYTE, i, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (task.id != EMPTY_QUEUE_RESPONSE) {
                pthread_mutex_lock(&mutex);
                task_queue_push(task_queue, task);
                pthread_mutex_unlock(&mutex);

                received_tasks++;
            }
        }

        if (received_tasks == 0) {
            pthread_mutex_lock(&mutex);
            termination = true;
            pthread_mutex_unlock(&mutex);
        }

        pthread_mutex_lock(&mutex);
        pthread_cond_signal(&worker_cond);
        pthread_mutex_unlock(&mutex);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Send(&termination_signal, 1, MPI_INT, process_id, REQUEST_TAG, MPI_COMM_WORLD);
    pthread_exit(NULL);
}

void* sender_start() {
    while (true) {
        int receive_process_id;

        Task task;

        MPI_Recv(&receive_process_id, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (receive_process_id == TERMINATION_SIGNAL) {
            break;
        }

        pthread_mutex_lock(&mutex);
        if (!task_queue_is_empty(task_queue)) {
            task_queue_pop(task_queue, &task);
        }
        else {
            task.id = EMPTY_QUEUE_RESPONSE;
            task.weight = 0;
            task.process_id = process_id;
        }
        pthread_mutex_unlock(&mutex);

        MPI_Send(&task, sizeof(task), MPI_BYTE, receive_process_id, RESPONSE_TAG, MPI_COMM_WORLD);
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    double start_time;
    double end_time;
    pthread_t worker_thread;
    pthread_t receiver_thread;
    pthread_t sender_thread;

    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided != required) {
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    task_queue = task_queue_create(TASK_COUNT);

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&worker_cond, NULL);
    pthread_cond_init(&receiver_cond, NULL);

    start_time = MPI_Wtime();
    pthread_create(&worker_thread, NULL, worker_start, NULL);
    pthread_create(&receiver_thread, NULL, receiver_start, NULL);
    pthread_create(&sender_thread, NULL, sender_start, NULL);

    pthread_join(worker_thread, NULL);
    pthread_join(receiver_thread, NULL);
    pthread_join(sender_thread, NULL);
    end_time = MPI_Wtime();

    double time = end_time - start_time;
    double finalTime = 0;
    MPI_Reduce(&time, &finalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Summary weight %d - start: %d, actual: %d\n", process_id, process_start_sum_weight, process_complete_tasks_weight_sum);
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 0) {
        printf("Time: %lf\n", finalTime);
    }

    task_queue_destroy(&task_queue);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&worker_cond);
    pthread_cond_destroy(&receiver_cond);
    MPI_Finalize();

    return EXIT_SUCCESS;
}