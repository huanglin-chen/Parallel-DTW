#include <cuda.h>
#include <algorithm>
#include "prefix.h"

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

void prefix_sequential(const double *X, const int start_row, const int end_row, const int n, const double *Y, const int m, double **D)
{
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
    double *x = (double *)malloc(m * sizeof(double));
    double *y = (double *)malloc(m * sizeof(double));
    for (int i = start_row; i < end_row; i++)
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
    free(x);
    free(y);
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

__global__ void dtw_prefix_kernel(const double *X, const int start_row, const int end_row, const double *Y, const int m, double *D, double *x, double *y)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_row; i < end_row; i++)
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

void prefix_parallel(const double *X, const int start_row, const int end_row, const int n, const double *Y, const int m, double **D)
{
    double *h_D;
    double *d_X, *d_Y, *d_D;
    double *d_x, *d_y;
    // memory initialization and memcpy
    h_D = (double *)malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        memcpy(&(h_D[i * m]), D[i], m * sizeof(double));
    }
    cudaMalloc((void **)&d_X, n * sizeof(double));
    cudaMalloc((void **)&d_Y, m * sizeof(double));
    cudaMalloc((void **)&d_D, n * m * sizeof(double));
    cudaMalloc((void **)&d_x, m * sizeof(double));
    cudaMalloc((void **)&d_y, m * sizeof(double));

    cudaMemcpy(d_X, X, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, n * m * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel function
    int numBlocks = 1;
    int threadsPerBlock = m;
    size_t shmemBytes = m * sizeof(double);
    dtw_prefix_kernel<<<numBlocks, threadsPerBlock, shmemBytes>>>(d_X, start_row, end_row, d_Y, m, d_D, d_x, d_y);

    // cuda synchronization
    cudaDeviceSynchronize();

    // memcpy and memory free
    cudaMemcpy(h_D, d_D, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
    {
        memcpy(D[i], &(h_D[i * m]), m * sizeof(double));
    }
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_D);
}

void prefix_sequential(const double *X, const int n, const double *Y, const int m, double **D)
{
    prefix_sequential(X, 1, n, n, Y, m, D);
}

void prefix_parallel(const double *X, const int n, const double *Y, const int m, double **D)
{
    prefix_parallel(X, 1, n, n, Y, m, D);
}
