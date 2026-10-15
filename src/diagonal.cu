#include <cuda.h>
#include <algorithm>
#include <vector>
#include "diagonal.h"

extern double *array_init_1d(const double *, const int, const double *, const int);
extern double **array_init_2d(const double *, const int, const double *, const int);

double **diagonal_sequential(const double *X, const int n, const double *Y, const int m)
{
    double **D = array_init_2d(X, n, Y, m);
    const int rows = n - 1;
    const int cols = m - 1;
    // diagonal: from top-right to down-left
    // d = i + j
    // d = 2 => (1,1)
    // d = 3 => (1,2), (2,1)
    // d = 4 => (1,3), (2,2), (3,1)
    for (int d = 2; d <= rows + cols; d++)
    {
        int i_start = std::max(1, d - cols);
        int i_end = std::min(rows, d - 1);

        for (int i = i_start; i <= i_end; i++)
        {
            int j = d - i;
            D[i][j] = ED(X[i], Y[j]) + std::min({D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]});
        }
    }
    return D;
}

__global__ void dtw_diag_kernel(const double *X, const int n, const double *Y, const int m, double *D)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = n - 1;
    const int cols = m - 1;

    for (int d = 2; d <= rows + cols; d++)
    {
        int i_start = max(1, d - cols);
        int i_end = min(rows, d - 1);
        int len = i_end - i_start + 1;

        if (id < len)
        {
            int i = i_start + id;
            int j = d - i;
            D[i * m + j] = ED(X[i], Y[j]) + min(D[(i - 1) * m + j], min(D[i * m + j - 1], D[(i - 1) * m + j - 1]));
        }
        __syncthreads();
    }
}

double **diagonal_parallel(const double *X, const int n, const double *Y, const int m)
{
    double *h_D = array_init_1d(X, n, Y, m);
    double *d_X;
    double *d_Y;
    double *d_D;
    cudaMalloc((void **)&d_X, n * sizeof(double));
    cudaMalloc((void **)&d_Y, m * sizeof(double));
    cudaMalloc((void **)&d_D, n * m * sizeof(double));
    cudaMemcpy(d_X, X, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, n * m * sizeof(double), cudaMemcpyHostToDevice);

    int block_num = 1;
    int thread_num = 1024;
    dtw_diag_kernel<<<block_num, thread_num>>>(d_X, n, d_Y, m, d_D);

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
    return D;
}

std::vector<float> diagonal_parallel_benchmark(const double *X, const int n, const double *Y, const int m)
{
    int rounds = 1000;
    int warmup = 10;
    int kernel_num = 1;
    std::vector<float> total_ms(kernel_num, 0);

    double *h_D = array_init_1d(X, n, Y, m);
    double *d_X;
    double *d_Y;
    double *d_D;
    cudaMalloc((void **)&d_X, n * sizeof(double));
    cudaMalloc((void **)&d_Y, m * sizeof(double));
    cudaMalloc((void **)&d_D, n * m * sizeof(double));
    cudaMemcpy(d_X, X, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, n * m * sizeof(double), cudaMemcpyHostToDevice);

    // warmup
    int block_num = 1;
    int thread_num = 1024;
    for (int round = 0; round < warmup; round++)
    {
        dtw_diag_kernel<<<block_num, thread_num>>>(d_X, n, d_Y, m, d_D);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for (int round = 0; round < rounds; round++)
    {
        cudaEventRecord(start);

        dtw_diag_kernel<<<block_num, thread_num>>>(d_X, n, d_Y, m, d_D);

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

    return total_ms;
}
