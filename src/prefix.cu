#include <cuda.h>
#include <algorithm>
#include <vector>
#include "prefix.h"

extern double *array_init_1d(const double *, const int, const double *, const int);
extern double **array_init_2d(const double *, const int, const double *, const int);

static void prefix_sum(double *seq, const int n)
{
    for (int i = 1; i < n; i++)
    {
        seq[i] += seq[i - 1];
    }
}

static void prefix_min(double *seq, const int n)
{
    for (int i = 1; i < n; i++)
    {
        seq[i] = min(seq[i], seq[i - 1]);
    }
}

void prefix_sequential(const double *X, const int row_start, const int row_end, const int n, const double *Y, const int m, double **D)
{
    double *x = new double[m];
    double *y = new double[m];
    /*
        t   = min(D[i-1][j], D[i-1][j-1]) + Dist[j]
            = min(D[i-1][j], D[i-1][j-1]) + (y[j]-y[j-1])
        z   = t - y[j]
            = min(D[i-1][j], D[i-1][j-1]) + (y[j]-y[j-1]) - y[j]
            = min(D[i-1][j], D[i-1][j-1]) - y[j-1]
        if z is assigned to x[j], then the equation
            x[j] = min(z, x[j-1])
        turns into
            x[j] = min(x[j], x[j-1])
    */
    for (int i = row_start; i < row_end; i++)
    {
        // calculate y
        for (int j = 0; j < m; j++)
        {
            y[j] = ED(X[i], Y[j]);
        }
        prefix_sum(y, m);
        // calculate x
        x[0] = D[i][0];
        for (int j = 1; j < m; j++)
        {
            x[j] = min(D[i - 1][j], D[i - 1][j - 1]) - y[j - 1];
        }
        prefix_min(x, m);
        // calculate D
        for (int j = 1; j < m; j++)
        {
            D[i][j] = x[j] + y[j];
        }
    }
    delete[] x;
    delete[] y;
}

/*
    The idea is to build a balanced binary tree on the input data and sweep it to and from the root to compute the prefix sum.
*/
__device__ void prefix_sum_parallel(double *seq, const int n)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double temp[];
    int offset = 1;

    // Phase1: up-sweep
    if (id < n)
    {
        temp[id] = seq[id];
    }

    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (id < d)
        {
            int ai = offset * (2 * id + 1) - 1;
            int bi = offset * (2 * id + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (id == 0)
    {
        temp[n - 1] = 0;
    }
    // Phase2: down-sweep
    for (int d = 1; d < n; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (id < d)
        {
            int ai = offset * (2 * id + 1) - 1;
            int bi = offset * (2 * id + 2) - 1;
            double t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (id < n - 1)
    {
        seq[id] = temp[id + 1];
    }
    else if (id == n - 1)
    {
        seq[id] = temp[id] + seq[id];
    }
}

__device__ void prefix_min_parallel(double *seq, const int n)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double temp[];
    int offset = 1;

    // Phase1: up-sweep
    if (id < n)
    {
        temp[id] = seq[id];
    }

    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (id < d)
        {
            int ai = offset * (2 * id + 1) - 1;
            int bi = offset * (2 * id + 2) - 1;
            temp[bi] = min(temp[ai], temp[bi]);
        }
        offset <<= 1;
    }

    if (id == 0)
    {
        temp[n - 1] = 1e308;
    }

    // Phase2: down-sweep
    for (int d = 1; d < n; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (id < d)
        {
            int ai = offset * (2 * id + 1) - 1;
            int bi = offset * (2 * id + 2) - 1;
            double t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = min(temp[bi], t);
        }
    }
    __syncthreads();

    if (id < n - 1)
    {
        seq[id] = temp[id + 1];
    }
    else if (id == n - 1)
    {
        seq[id] = min(seq[id], temp[id]);
    }
}

__global__ void dtw_prefix_kernel(const double *X, const int row_start, const int row_end, const double *Y, const int m, double *D, double *x, double *y)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = row_start; i < row_end; i++)
    {
        // calculate y
        if (id < m)
        {
            y[id] = ED(X[i], Y[id]);
        }
        if (id == 0)
        {
            x[id] = D[i * m + id];
        }
        __syncthreads();

        prefix_sum_parallel(y, m);

        // calculate x
        if (1 <= id && id < m)
        {
            x[id] = min(D[(i - 1) * m + id], D[(i - 1) * m + (id - 1)]) - y[id - 1];
        }
        __syncthreads();

        prefix_min_parallel(x, m);

        // calculate D
        if (1 <= id && id < m)
        {
            D[i * m + id] = x[id] + y[id];
        }
        __syncthreads();
    }
}

double **prefix_parallel(const double *X, const int row_start, const int row_end, const int n, const double *Y, const int m)
{
    double *h_D = array_init_1d(X, n, Y, m);
    double *d_X;
    double *d_Y;
    double *d_D;
    double *d_x;
    double *d_y;
    cudaMalloc((void **)&d_X, n * sizeof(double));
    cudaMalloc((void **)&d_Y, m * sizeof(double));
    cudaMalloc((void **)&d_D, n * m * sizeof(double));
    cudaMalloc((void **)&d_x, m * sizeof(double));
    cudaMalloc((void **)&d_y, m * sizeof(double));
    cudaMemcpy(d_X, X, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, n * m * sizeof(double), cudaMemcpyHostToDevice);

    int block_num = 1;
    int thread_num = m;
    size_t shm_byte = m * sizeof(double);
    dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, row_end, d_Y, m, d_D, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(h_D, d_D, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    double **D = new double *[n];
    for (int i = 0; i < n; i++)
    {
        D[i] = new double[m];
        memcpy(D[i], &(h_D[i * m]), m * sizeof(double));
    }
    delete[] h_D;
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    return D;
}

std::vector<float> prefix_parallel_benchmark(const double *X, const int row_start, const int row_end, const int n, const double *Y, const int m)
{
    int rounds = 100;
    int warmup = 10;
    int kernel_num = 1;
    std::vector<float> total_ms(kernel_num, 0);

    double *h_D = array_init_1d(X, n, Y, m);
    double *d_X;
    double *d_Y;
    double *d_D;
    double *d_x;
    double *d_y;
    cudaMalloc((void **)&d_X, n * sizeof(double));
    cudaMalloc((void **)&d_Y, m * sizeof(double));
    cudaMalloc((void **)&d_D, n * m * sizeof(double));
    cudaMalloc((void **)&d_x, m * sizeof(double));
    cudaMalloc((void **)&d_y, m * sizeof(double));
    cudaMemcpy(d_X, X, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, n * m * sizeof(double), cudaMemcpyHostToDevice);

    // warmup
    int block_num = 1;
    int thread_num = m;
    size_t shm_byte = m * sizeof(double);
    for (int round = 0; round < warmup; round++)
    {
        dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, row_end, d_Y, m, d_D, d_x, d_y);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for (int round = 0; round < rounds; round++)
    {
        cudaEventRecord(start);

        dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, row_end, d_Y, m, d_D, d_x, d_y);

        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start, end);
        total_ms[0] += ms;
    }
    for (int i = 0; i < kernel_num; i++)
    {
        total_ms[i] /= rounds;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Skip copying data back
    delete[] h_D;
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    return total_ms;
}

double **prefix_sequential(const double *X, const int n, const double *Y, const int m)
{
    double **D = array_init_2d(X, n, Y, m);
    prefix_sequential(X, 1, n, n, Y, m, D);
    return D;
}

double **prefix_parallel(const double *X, const int n, const double *Y, const int m)
{
    return prefix_parallel(X, 1, n, n, Y, m);
}

std::vector<float> prefix_parallel_benchmark(const double *X, const int n, const double *Y, const int m)
{
    return prefix_parallel_benchmark(X, 1, n, n, Y, m);
}
