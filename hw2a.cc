// #define DEBUG
// #define TIMING
/*
 * 0: static
 * 1: dynamic
 * 2: guided
 */
#define SCHEDULE 1
/**
 * 0: By index
 * 1: By row
 * 2: By column
 * 3: By block
 */
#define PARTITION 1
#define VECTORIZATION
// #undef VECTORIZATION

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <immintrin.h>
#include <math.h>
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
#else
#define DEBUG_PRINT(fmt, args...)
#endif

#ifdef TIMING
#include <time.h>
clock_t __start_time, __tot_start_time;
double __duration, __tot_duration;
#define TIMING_START() \
    __start_time = clock();
#define TIMING_END(arg)                                             \
    __duration = (double)(clock() - __start_time) / CLOCKS_PER_SEC; \
    DEBUG_PRINT("%s, %lf\n", arg, __duration);
#define TOT_TIMING_START() \
    __tot_start_time = clock();
#define TOT_TIMING_END()                                                    \
    __tot_duration = (double)(clock() - __tot_start_time) / CLOCKS_PER_SEC; \
    DEBUG_PRINT("Total, %lf\n", __tot_duration);
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
#elif PARTITION == 2
    int start;
    int end;
#elif PARTITION == 3
    int start_i;
    int start_j;
    int end_i;
    int end_j;
#endif  // PARTITION
} Task;
typedef struct TaskPool {
#if PARTITION == 0
    int taskId;
    int chunk;
#elif PARTITION == 1
    int taskId;
    int chunk;
#elif PARTITION == 2
    int taskId;
    int chunk;
#elif PARTITION == 3
    int task_i;
    int task_j;
    int chunk_i;
    int chunk_j;
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
} SharedData;
typedef struct LocalData {
    size_t tid;
} LocalData;
typedef struct Data {
    SharedData* sharedData;
    LocalData* localData;
} Data;

/* Utilities */
int min(int x, int y);

/* Gets taskid with locking */
Task get_task(Data* data);
void* func(Data* data);
void* thread_func(void* arg);

void write_png(const char* filename, int iters, int width, int height, const int* buffer);

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    DEBUG_PRINT("%d cpus available\n", CPU_COUNT(&cpu_set));
    TOT_TIMING_START();

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
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* pthread */
    size_t ncpus = CPU_COUNT(&cpu_set);
    TaskPool taskPool;
    SharedData sharedData;
    LocalData* localData = (LocalData*)malloc(ncpus * sizeof(LocalData));
    Data* data = (Data*)malloc(ncpus * sizeof(Data));
    pthread_t* threadId = (pthread_t*)malloc(ncpus * sizeof(pthread_t));

    sharedData.iters = iters;
    sharedData.left = left;
    sharedData.right = right;
    sharedData.lower = lower;
    sharedData.upper = upper;
    sharedData.width = width;
    sharedData.height = height;
    sharedData.image = image;
    sharedData.taskPool = &taskPool;

#if PARTITION == 0
    taskPool.taskId = 0;
#if SCHEDULE == 0
    taskPool.chunk = ceil((double)(width * height) / ncpus);
#elif SCHEDULE == 1
    // taskPool.chunk = ceil((double)(width * height) / 10000);
    taskPool.chunk = 3 * width;
#endif  // SCHEDULE
#elif PARTITION == 1
    taskPool.taskId = 0;
    taskPool.chunk = 1;
#elif PARTITION == 2
    taskPool.taskId = 0;
    taskPool.chunk = 1;
#elif PARTITION == 3
    taskPool.task_i = 0;
    taskPool.task_j = 0;
    taskPool.chunk_i = 50;
    taskPool.chunk_j = 10;
#endif  // PARTITION

    pthread_mutex_init(&taskPool.mutex, NULL);

    for (size_t tid = 0; tid < ncpus; tid++) {
        localData[tid].tid = tid;
        data[tid].sharedData = &sharedData;
        data[tid].localData = &localData[tid];
    }

    /* Logging */
    DEBUG_PRINT("Image size: %d\n", width * height);
    DEBUG_PRINT("Chunk: %d\n", taskPool.chunk);
#if SCHEUDLE == 2
    DEBUG_PRINT("Decrement: %d\n", taskPool.decrement);
#endif

    /* pthread mandelbrot set */
    for (size_t tid = 1; tid < ncpus; tid++) {
        pthread_create(&threadId[tid], NULL, thread_func, &data[tid]);
    }
    func(&data[0]);
    for (size_t tid = 1; tid < ncpus; tid++) {
        pthread_join(threadId[tid], NULL);
    }

    /* draw and cleanup */
    TIMING_START()
    write_png(filename, iters, width, height, image);
    TIMING_END("write_png");
    free(image);
    free(localData);
    free(data);
    free(threadId);
    pthread_mutex_destroy(&taskPool.mutex);
    TOT_TIMING_END();
}

int min(int x, int y) {
    return (x < y) ? x : y;
}

Task get_task(Data* data) {
    LocalData* localData = data->localData;
    SharedData* sharedData = data->sharedData;
    TaskPool* taskPool = sharedData->taskPool;

    Task task;
    pthread_mutex_lock(&taskPool->mutex);
#if PARTITION == 0
    task.start = taskPool->taskId;
    taskPool->taskId += taskPool->chunk;
    task.end = taskPool->taskId;
#elif PARTITION == 1
    task.start = taskPool->taskId;
    taskPool->taskId += taskPool->chunk;
    task.end = taskPool->taskId;
#elif PARTITION == 2
    task.start = taskPool->taskId;
    taskPool->taskId += taskPool->chunk;
    task.end = taskPool->taskId;
#elif PARTITION == 3
    task.start_i = taskPool->task_i;
    task.start_j = taskPool->task_j;
    task.end_i = min(task.start_i + taskPool->chunk_i, sharedData->width);
    task.end_j = min(task.start_j + taskPool->chunk_j, sharedData->height);
    taskPool->task_i += taskPool->chunk_i;
    if (taskPool->task_i >= sharedData->width) {
        taskPool->task_i = 0;
        taskPool->task_j += taskPool->chunk_j;
    }
#endif  // PARTITION
    pthread_mutex_unlock(&taskPool->mutex);
    return task;
}
void* func(Data* data) {
    LocalData* localData = data->localData;
    SharedData* sharedData = data->sharedData;

    int iters = sharedData->iters;
    double left = sharedData->left;
    double right = sharedData->right;
    double lower = sharedData->lower;
    double upper = sharedData->upper;
    int width = sharedData->width;
    int height = sharedData->height;
    int* image = sharedData->image;

#if PARTITION == 0
    while (1) {
        Task task = get_task(data);
        if (task.start >= height * width)
            break;
#ifdef VECTORIZATION
        int id;
        for (id = task.start; id + 1 < task.end && id + 1 < height * width; id += 2) {
            __m128d vec_y0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd((id + 1) / width, id / width), _mm_set1_pd((upper - lower) / height)), _mm_set1_pd(lower));
            __m128d vec_x0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd((id + 1) % width, id % width), _mm_set1_pd((right - left) / width)), _mm_set1_pd(left));
            int repeats = 0;
            __m128i vec_repeat = _mm_setzero_si128();
            __m128d vec_x = _mm_setzero_pd();
            __m128d vec_y = _mm_setzero_pd();
            __m128d vec_length_squared = _mm_setzero_pd();
            __m128d vec_length_squared_threshold = _mm_set1_pd(4);
            while (repeats < iters) {
                __m128d vec_cmp = _mm_cmpgt_pd(vec_length_squared_threshold, vec_length_squared);
                if (_mm_movemask_pd(vec_cmp) == 0)
                    break;
                __m128d vec_temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_x0);
                vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(2), vec_x), vec_y), vec_y0);
                vec_x = vec_temp;
                vec_length_squared = _mm_blendv_pd(vec_length_squared, _mm_add_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_cmp);
                vec_repeat = _mm_add_epi64(vec_repeat, _mm_srli_epi64(_mm_castpd_si128(vec_cmp), 63));
                ++repeats;
            }
            long long repeat_arr[2];
            _mm_store_si128((__m128i*)repeat_arr, vec_repeat);
            image[id] = repeat_arr[0];
            image[id + 1] = repeat_arr[1];
        }
        if (id < task.end && id < height * width) {
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
#else
        for (int id = task.start; id < task.end && id < height * width; id++) {
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
            image[j * width + i] = repeats;
        }
#endif  // VECTORIZATION
    }
#elif PARTITION == 1
    while (1) {
        Task task = get_task(data);
        if (task.start >= height)
            break;
#ifdef VECTORIZATION
        for (int j = task.start, i; j < task.end && j < height; j++) {
            for (i = 0; i + 1 < width; i += 2) {
                __m128d vec_y0 = _mm_set1_pd(j * ((upper - lower) / height) + lower);
                __m128d vec_x0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd(i + 1, i), _mm_set1_pd((right - left) / width)), _mm_set1_pd(left));
                int repeats = 0;
                __m128i vec_repeat = _mm_setzero_si128();
                __m128d vec_x = _mm_setzero_pd();
                __m128d vec_y = _mm_setzero_pd();
                __m128d vec_length_squared = _mm_setzero_pd();
                __m128d vec_length_squared_threshold = _mm_set1_pd(4);
                while (repeats < iters) {
                    __m128d vec_cmp = _mm_cmpgt_pd(vec_length_squared_threshold, vec_length_squared);
                    if (_mm_movemask_pd(vec_cmp) == 0)
                        break;
                    __m128d vec_temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_x0);
                    vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(2), vec_x), vec_y), vec_y0);
                    vec_x = vec_temp;
                    vec_length_squared = _mm_blendv_pd(vec_length_squared, _mm_add_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_cmp);
                    vec_repeat = _mm_add_epi64(vec_repeat, _mm_srli_epi64(_mm_castpd_si128(vec_cmp), 63));
                    ++repeats;
                }
                long long repeat_arr[2];
                _mm_store_si128((__m128i*)repeat_arr, vec_repeat);
                image[j * width + i] = repeat_arr[0];
                image[j * width + i + 1] = repeat_arr[1];
            }
            if (i < width) {
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
#else
        for (int j = task.start; j < task.end && j < height; j++) {
            for (int i = 0; i < width; i++) {
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
#elif PARTITION == 2
    while (1) {
        Task task = get_task(data);
        if (task.start >= width)
            break;
#ifdef VECTORIZATION
        for (int i = task.start, j; i < task.end && i < width; i++) {
            for (j = 0; j + 1 < height; j += 2) {
                __m128d vec_y0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd(j + 1, j), _mm_set1_pd((upper - lower) / height)), _mm_set1_pd(lower));
                __m128d vec_x0 = _mm_set1_pd(i * ((right - left) / width) + left);
                int repeats = 0;
                __m128i vec_repeat = _mm_setzero_si128();
                __m128d vec_x = _mm_setzero_pd();
                __m128d vec_y = _mm_setzero_pd();
                __m128d vec_length_squared = _mm_setzero_pd();
                __m128d vec_length_squared_threshold = _mm_set1_pd(4);
                while (repeats < iters) {
                    __m128d vec_cmp = _mm_cmpgt_pd(vec_length_squared_threshold, vec_length_squared);
                    if (_mm_movemask_pd(vec_cmp) == 0)
                        break;
                    __m128d vec_temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_x0);
                    vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(2), vec_x), vec_y), vec_y0);
                    vec_x = vec_temp;
                    vec_length_squared = _mm_blendv_pd(vec_length_squared, _mm_add_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_cmp);
                    vec_repeat = _mm_add_epi64(vec_repeat, _mm_srli_epi64(_mm_castpd_si128(vec_cmp), 63));
                    ++repeats;
                }
                long long repeat_arr[2];
                _mm_store_si128((__m128i*)repeat_arr, vec_repeat);
                image[j * width + i] = repeat_arr[0];
                image[(j + 1) * width + i] = repeat_arr[1];
            }
            if (j < height) {
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
#else
        for (int i = task.start; i < task.end && i < width; i++) {
            for (int j = 0; j < height; j++) {
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
#elif PARTITION == 3
    double coef_j = (upper - lower) / height;
    double coef_i = (right - left) / width;
    while (1) {
        Task task = get_task(data);
        if (task.start_j >= height)
            break;
#ifdef VECTORIZATION
        for (int j = task.start_j, i; j < task.end_j; j++) {
            for (i = task.start_i; i + 1 < task.end_i; i += 2) {
                __m128d vec_y0 = _mm_set1_pd(j * ((upper - lower) / height) + lower);
                __m128d vec_x0 = _mm_add_pd(_mm_mul_pd(_mm_set_pd(i + 1, i), _mm_set1_pd((right - left) / width)), _mm_set1_pd(left));
                int repeats = 0;
                __m128i vec_repeat = _mm_setzero_si128();
                __m128d vec_x = _mm_setzero_pd();
                __m128d vec_y = _mm_setzero_pd();
                __m128d vec_length_squared = _mm_setzero_pd();
                __m128d vec_length_squared_threshold = _mm_set1_pd(4);
                while (repeats < iters) {
                    __m128d vec_cmp = _mm_cmpgt_pd(vec_length_squared_threshold, vec_length_squared);
                    if (_mm_movemask_pd(vec_cmp) == 0)
                        break;
                    __m128d vec_temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_x0);
                    vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(2), vec_x), vec_y), vec_y0);
                    vec_x = vec_temp;
                    vec_length_squared = _mm_blendv_pd(vec_length_squared, _mm_add_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), vec_cmp);
                    vec_repeat = _mm_add_epi64(vec_repeat, _mm_srli_epi64(_mm_castpd_si128(vec_cmp), 63));
                    ++repeats;
                }
                long long repeat_arr[2];
                _mm_store_si128((__m128i*)repeat_arr, vec_repeat);
                image[j * width + i] = repeat_arr[0];
                image[j * width + i + 1] = repeat_arr[1];
            }
            if (i < task.end_i) {
                double y0 = j * coef_j + lower;
                double x0 = i * coef_i + left;
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
#else
        for (int j = task.start_j; j < task.end_j; j++) {
            for (int i = task.start_i; i < task.end_i; i++) {
                double y0 = j * coef_j + lower;
                double x0 = i * coef_i + left;
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
void* thread_func(void* arg) {
    Data* data = (Data*)arg;
    void* retVal = func(data);
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