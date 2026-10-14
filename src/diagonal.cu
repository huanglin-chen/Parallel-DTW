#include <cuda.h>
#include <algorithm>
#include "diagonal.h"

void diagonal_sequential(const double *X, const int n, const double *Y, const int m, double **D)
{
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

void diagonal_parallel(const double *X, const int n, const double *Y, const int m, double **D)
{
    // memory initialization and memcpy
    double *d_X;
    double *d_Y;
    double *d_D;
    double *h_D;
    h_D = (double *)malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        memcpy(&(h_D[i * m]), D[i], m * sizeof(double));
    }
    cudaMalloc((void **)&d_X, n * sizeof(double));
    cudaMalloc((void **)&d_Y, m * sizeof(double));
    cudaMalloc((void **)&d_D, n * m * sizeof(double));
    cudaMemcpy(d_X, X, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, n * m * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel function
    int numBlocks = 1;
    int threadsPerBlock = 1024;
    dtw_diag_kernel<<<numBlocks, threadsPerBlock>>>(d_X, n, d_Y, m, d_D);

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
    free(h_D);
}
