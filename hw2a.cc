// #define DEBUG
// #define TIMING
/*
0: static
1: dynamic
2: guided
 */
#define SCHEDULE 1

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
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
    int start;
    int end;
} Task;
typedef struct TaskPool {
    int taskId;
    int chunk;
#if SCHEDULE == 2
    int decrement;
#endif
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

/* Gets taskid with locking */
Task get_task(TaskPool* taskPool);
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

    taskPool.taskId = 0;
#if SCHEDULE == 0
    taskPool.chunk = ceil((double)(width * height) / ncpus);
#elif SCHEDULE == 1
    // taskPool.chunk = ceil((double)(width * height) / 10000);
    taskPool.chunk = 1000;
#elif SCHEDULE == 2
    taskPool.chunk = ceil((double)(width * height) / 1000);
    taskPool.decrement = 100;
#endif
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

Task get_task(TaskPool* taskPool) {
    Task task;
    pthread_mutex_lock(&taskPool->mutex);
    task.start = taskPool->taskId;
    taskPool->taskId += taskPool->chunk;
    task.end = taskPool->taskId;
#if SCHEDULE == 2
    if (taskPool->chunk > taskPool->decrement)
        taskPool->chunk -= taskPool->decrement;
    else
        taskPool->chunk = 1;
#endif
    pthread_mutex_unlock(&taskPool->mutex);
    return task;
}
void* func(Data* data) {
    LocalData* localData = data->localData;
    SharedData* sharedData = data->sharedData;
    TaskPool* taskPool = sharedData->taskPool;

    int iters = sharedData->iters;
    double left = sharedData->left;
    double right = sharedData->right;
    double lower = sharedData->lower;
    double upper = sharedData->upper;
    int width = sharedData->width;
    int height = sharedData->height;
    int* image = sharedData->image;

    while (1) {
        Task task = get_task(taskPool);
        if (task.start >= height * width)
            break;
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
    }
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