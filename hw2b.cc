// #define DEBUG
// #define TIMING
/**
 * 0: By index
 * 1: By row
 */
#define PARTITION 0
#define VECTORIZATION
// #undef VECTORIZATION

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...)     \
    do {                              \
        fprintf(stderr, fmt, ##args); \
    } while (false);
#define MPI_EXECUTE(func)                                          \
    {                                                              \
        int rc = func;                                             \
        if (rc != MPI_SUCCESS) {                                   \
            printf("Error on MPI function at line %d.", __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, rc);                         \
        }                                                          \
    }
#else
#define DEBUG_PRINT(fmt, args...)
#define MPI_EXECUTE(func) func
#endif

#ifdef TIMING
double __start_time, __duration, __sum_duration, __max_duration, __min_duration;
double __tot_start_time, __tot_duration, __sum_tot_duration, __max_tot_duration, __min_tot_duration;
#define TIMING_START() \
    __start_time = MPI_Wtime();
#define TIMING_END(arg)                                                                                       \
    __duration = MPI_Wtime() - __start_time;                                                                  \
    MPI_Reduce(&__duration, &__sum_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                      \
    MPI_Reduce(&__duration, &__max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                      \
    MPI_Reduce(&__duration, &__min_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                      \
    if (world_rank == 0) {                                                                                    \
        DEBUG_PRINT("%s, %lf, %lf, %lf\n", arg, __sum_duration / world_size, __max_duration, __min_duration); \
    }
#define TOT_TIMING_START() \
    __tot_start_time = MPI_Wtime();
#define TOT_TIMING_END()                                                                                                \
    __tot_duration = MPI_Wtime() - __tot_start_time;                                                                    \
    MPI_Reduce(&__tot_duration, &__sum_tot_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                        \
    MPI_Reduce(&__tot_duration, &__max_tot_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                        \
    MPI_Reduce(&__tot_duration, &__min_tot_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                        \
    if (world_rank == 0) {                                                                                              \
        DEBUG_PRINT("Total, %lf, %lf, %lf\n", __sum_tot_duration / world_size, __max_tot_duration, __min_tot_duration); \
    }
#else
#define TIMING_START()
#define TIMING_END(arg)
#define TOT_TIMING_START()
#define TOT_TIMING_END()
#endif

typedef struct Task {
#if PARTITION == 0
    int start;
    int end;
#elif PARTITION == 1
    int start;
    int end;
#endif  // PARTITION
} Task;
typedef struct TaskPool {
#if PARTITION == 0
    int taskId;
    int chunk;
#elif PARTITION == 1
    int taskId;
    int chunk;
#endif  // PARTITION
    pthread_mutex_t mutex;
} TaskPool;
typedef struct SharedData {
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
    int* image;
    TaskPool* taskPool;
    int world_size;
    int world_rank;
    size_t ncpus;
} SharedData;
typedef struct RankData {
} RankData;

typedef struct Data {
    SharedData* sharedData;
} Data;

/* Utilities */
int min(int x, int y);
int max(int x, int y);

/* Gets taskid with locking */
Task get_task(Data* data);

/* MPI Communicating */
void* controller(Data*);
void* worker(Data*);
void* thread_controller(void*);
void* thread_worker(void*);

void write_png(const char* filename, int iters, int width, int height, const int* buffer);

int main(int argc, char** argv) {
    int mpi_thread_supported;
    MPI_EXECUTE(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_supported));
    assert(mpi_thread_supported == MPI_THREAD_MULTIPLE);

    int world_rank, world_size;
    MPI_EXECUTE(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_EXECUTE(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    TOT_TIMING_START();
    TIMING_START();
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    DEBUG_PRINT("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* tot_image = (int*)malloc(width * height * sizeof(int));
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    memset(image, 0, width * height * sizeof(int));

    /* MPI */
    TaskPool taskPool;
    SharedData sharedData;
    Data data;
    pthread_t thread;

    sharedData.iters = iters;
    sharedData.left = left;
    sharedData.right = right;
    sharedData.lower = lower;
    sharedData.upper = upper;
    sharedData.width = width;
    sharedData.height = height;
    sharedData.image = image;
    sharedData.taskPool = &taskPool;
    sharedData.world_size = world_size;
    sharedData.world_rank = world_rank;
    sharedData.ncpus = CPU_COUNT(&cpu_set);

#if PARTITION == 0
    taskPool.taskId = 0;
    taskPool.chunk = width * sharedData.ncpus;
#elif PARTITION == 1
    taskPool.taskId = 0;
    taskPool.chunk = sharedData.ncpus;
#endif  // PARTITION

    data.sharedData = &sharedData;

    TIMING_END("Initialize");
    TIMING_START();
    if (world_size > 1 && world_rank == 0) {
        sharedData.ncpus--;
        pthread_mutex_init(&taskPool.mutex, NULL);
        pthread_create(&thread, NULL, thread_controller, &data);
    }

    if (world_rank < world_size && sharedData.ncpus) {
        worker(&data);
    }

    if (world_size > 1 && world_rank == 0) {
        pthread_join(thread, NULL);
        pthread_mutex_destroy(&taskPool.mutex);
    }
    TIMING_END("Mandlebrot");

    TIMING_START();
    MPI_Reduce(image, tot_image, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    TIMING_END("Reduce");

    /* draw */
    if (world_rank == 0) {
#ifdef TIMING
        double wp_stime = MPI_Wtime();
#endif
        write_png(filename, iters, width, height, tot_image);
#ifdef TIMING
        double wp_duration = MPI_Wtime() - wp_stime;
        DEBUG_PRINT("write_png: %lf\n", wp_duration);
#endif
    }
    TOT_TIMING_END();

    free(tot_image);
    free(image);
    MPI_Finalize();
}

int min(int x, int y) {
    return (x < y) ? x : y;
}
int max(int x, int y) {
    return (x > y) ? x : y;
}

Task get_task(Data* data) {
    SharedData* sharedData = data->sharedData;
    TaskPool* taskPool = sharedData->taskPool;

    Task task;
    if (sharedData->world_rank == 0) {
        pthread_mutex_lock(&taskPool->mutex);
#if PARTITION == 0
        task.start = taskPool->taskId;
        taskPool->taskId += taskPool->chunk;
        task.end = taskPool->taskId;
#elif PARTITION == 1
        task.start = taskPool->taskId;
        taskPool->taskId += taskPool->chunk;
        task.end = taskPool->taskId;
#endif  // PARTITION
        pthread_mutex_unlock(&taskPool->mutex);
    } else {
        pthread_mutex_lock(&taskPool->mutex);
        MPI_Send(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&task, sizeof(Task) / sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        pthread_mutex_unlock(&taskPool->mutex);
    }
    return task;
}
void* controller(Data* data) {
    SharedData* sharedData = data->sharedData;
    TaskPool* taskPool = sharedData->taskPool;
    assert(sharedData->world_rank == 0);

    MPI_Status status;
    MPI_Request request;
    Task task;
    int done_cnt = 0;
    while (done_cnt < sharedData->world_size - 1) {
        MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

#if PARTITION == 0
        if (taskPool->taskId >= sharedData->height * sharedData->width)
            done_cnt++;
        pthread_mutex_lock(&taskPool->mutex);
        task.start = taskPool->taskId;
        taskPool->taskId += taskPool->chunk;
        task.end = taskPool->taskId;
        pthread_mutex_unlock(&taskPool->mutex);
#elif PARTITION == 1
        if (taskPool->taskId >= sharedData->height)
            done_cnt++;
        pthread_mutex_lock(&taskPool->mutex);
        task.start = taskPool->taskId;
        taskPool->taskId += taskPool->chunk;
        task.end = taskPool->taskId;
        pthread_mutex_unlock(&taskPool->mutex);
#endif  // PARTITION
        MPI_Isend(&task, sizeof(Task) / sizeof(int), MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
    }
    return NULL;
}
void* worker(Data* data) {
    SharedData* sharedData = data->sharedData;

    int iters = sharedData->iters;
    double left = sharedData->left;
    double right = sharedData->right;
    double lower = sharedData->lower;
    double upper = sharedData->upper;
    int width = sharedData->width;
    int height = sharedData->height;
    int* image = sharedData->image;
    size_t ncpus = sharedData->ncpus;

#ifdef VECTORIZATION
    __m128d vec_d_4 = _mm_set1_pd(4);
    __m128d vec_d_2 = _mm_set1_pd(2);
    __m128d vec_ulh = _mm_set1_pd((upper - lower) / height);
    __m128d vec_rlw = _mm_set1_pd((right - left) / width);
    __m128d vec_lower = _mm_set1_pd(lower);
    __m128d vec_left = _mm_set1_pd(left);
#endif  // VECTORIZATION

#if PARTITION == 0
    while (1) {
        Task task = get_task(data);
        if (task.start >= height * width)
            break;
        int id, id_offset;
        int id_end = min(task.end, height * width);
        int id_offset_end = id_end - task.start;
#ifdef VECTORIZATION
#pragma omp parallel for num_threads(ncpus) schedule(dynamic) default(shared)
        for (id_offset = 0; id_offset < id_offset_end; id_offset += 2) {
            id = task.start + id_offset;
            if (id + 1 < id_end) {
                __m128d vec_y0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd((id + 1) / width, id / width), vec_ulh), vec_lower);
                __m128d vec_x0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd((id + 1) % width, id % width), vec_rlw), vec_left);
                int repeats = 0;
                __m128i vec_repeat = _mm_setzero_si128();
                __m128d vec_x = _mm_setzero_pd(), vec_x_sq = _mm_setzero_pd();
                __m128d vec_y = _mm_setzero_pd(), vec_y_sq = _mm_setzero_pd();
                __m128d vec_length_squared = _mm_setzero_pd();
                while (repeats < iters) {
                    __m128d vec_cmp = _mm_cmpgt_pd(vec_d_4, vec_length_squared);
                    if (_mm_movemask_pd(vec_cmp) == 0)
                        break;
                    __m128d vec_temp = _mm_add_pd(_mm_sub_pd(vec_x_sq, vec_y_sq), vec_x0);
                    vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(vec_d_2, vec_x), vec_y), vec_y0);
                    vec_x = vec_temp;
                    vec_x_sq = _mm_mul_pd(vec_x, vec_x);
                    vec_y_sq = _mm_mul_pd(vec_y, vec_y);
                    vec_length_squared = _mm_blendv_pd(vec_length_squared, _mm_add_pd(vec_x_sq, vec_y_sq), vec_cmp);
                    vec_repeat = _mm_add_epi64(vec_repeat, _mm_srli_epi64(_mm_castpd_si128(vec_cmp), 63));
                    ++repeats;
                }
                _mm_storel_epi64((__m128i*)(image + id), _mm_shuffle_epi32(vec_repeat, 0b01000));
            } else {
                int j = id / width;
                int i = id % width;
                double y0 = j * ((upper - lower) / height) + lower;
                double x0 = i * ((right - left) / width) + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[id] = repeats;
            }
        }
#else
#pragma omp parallel for num_threads(ncpus) schedule(dynamic) default(shared)
        for (id_offset = 0; id_offset < id_offset_end; id_offset++) {
            id = task.start + id_offset;
            int j = id / width;
            int i = id % width;
            double y0 = j * ((upper - lower) / height) + lower;
            double x0 = i * ((right - left) / width) + left;
            int repeats = 0;
            double x = 0, x_sq = 0;
            double y = 0, y_sq = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x_sq - y_sq + x0;
                y = 2 * x * y + y0;
                x = temp;
                x_sq = x * x;
                y_sq = y * y;
                length_squared = x_sq + y_sq;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
#endif  // VECTORIZATION
    }
#elif PARTITION == 1
    while (1) {
        Task task = get_task(data);
        if (task.start >= height)
            break;
        int j_shift_end = min(task.end, height) - task.start;
        int j_shift, j, i;
#ifdef VECTORIZATION
#pragma omp parallel for num_threads(ncpus) schedule(dynamic) default(shared) collapse(2)
        for (j_shift = 0; j_shift < j_shift_end; j_shift++) {
            for (i = 0; i < width; i += 2) {
                j = j_shift + task.start;
                if (i + 1 < width) {
                    __m128d vec_y0 = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(j), vec_ulh), vec_lower);
                    __m128d vec_x0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd(i + 1, i), vec_rlw), vec_left);
                    int repeats = 0;
                    __m128i vec_repeat = _mm_setzero_si128();
                    __m128d vec_x = _mm_setzero_pd(), vec_x_sq = _mm_setzero_pd();
                    __m128d vec_y = _mm_setzero_pd(), vec_y_sq = _mm_setzero_pd();
                    __m128d vec_length_squared = _mm_setzero_pd();
                    while (repeats < iters) {
                        __m128d vec_cmp = _mm_cmpgt_pd(vec_d_4, vec_length_squared);
                        if (_mm_movemask_pd(vec_cmp) == 0)
                            break;
                        __m128d vec_temp = _mm_add_pd(_mm_sub_pd(vec_x_sq, vec_y_sq), vec_x0);
                        vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(vec_d_2, vec_x), vec_y), vec_y0);
                        vec_x = vec_temp;
                        vec_x_sq = _mm_mul_pd(vec_x, vec_x);
                        vec_y_sq = _mm_mul_pd(vec_y, vec_y);
                        vec_length_squared = _mm_blendv_pd(vec_length_squared, _mm_add_pd(vec_x_sq, vec_y_sq), vec_cmp);
                        vec_repeat = _mm_add_epi64(vec_repeat, _mm_srli_epi64(_mm_castpd_si128(vec_cmp), 63));
                        ++repeats;
                    }
                    _mm_storel_epi64((__m128i*)(image + j * width + i), _mm_shuffle_epi32(vec_repeat, 0b01000));
                } else {
                    double y0 = j * ((upper - lower) / height) + lower;
                    double x0 = i * ((right - left) / width) + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[j * width + i] = repeats;
                }
            }
        }
#else
#pragma omp parallel for num_threads(ncpus) schedule(dynamic) default(shared) collapse(2)
        for (j_shift = 0; j_shift < j_shift_end; j_shift++) {
            for (i = 0; i < width; i += 2) {
                j = j_shift + task.start;
                double y0 = j * ((upper - lower) / height) + lower;
                double x0 = i * ((right - left) / width) + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[j * width + i] = repeats;
            }
        }
#endif  // VECTORIZATION
    }
#endif  // PARTITION
    return NULL;
}
void* thread_controller(void* arg) {
    Data* data = (Data*)arg;
    void* retVal = controller(data);
    pthread_exit(retVal);
}
void* thread_worker(void* arg) {
    Data* data = (Data*)arg;
    void* retVal = worker(data);
    pthread_exit(retVal);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
#pragma omp parallel for schedule(dynamic) default(shared)
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}