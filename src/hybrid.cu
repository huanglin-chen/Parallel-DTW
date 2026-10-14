#include <cuda.h>
#include <algorithm>
#include "hybrid.h"

extern void prefix_sequential(const double *, const int, const int, const int, const double *, const int, double **);
extern __global__ void dtw_prefix_kernel(const double *, const int, const int, const double *, const int, double *, double *, double *);

void hybrid_sequential(const double *X, const int n, const double *Y, const int m, double **D)
{
    // Diagonal section
    const int row_start = 1;
    const int row_end = WIDTH * ((n - 1) / WIDTH);

    int i[WIDTH];
    int j[WIDTH];
    for (int k = 0; k < WIDTH; k++)
    {
        i[k] = 1 + k;
        j[k] = 1 - k;
    }

    const int diag_num = (m - 1) * ((n - 1) / WIDTH) + WIDTH - 1;
    for (int diag = 0; diag < diag_num; diag++)
    {
        for (int k = 0; k < WIDTH; k++)
        {
            if (row_start <= j[k] && i[k] <= row_end)
            {
                D[i[k]][j[k]] = ED(X[i[k]], Y[j[k]]) + std::min({D[i[k] - 1][j[k]], D[i[k]][j[k] - 1], D[i[k] - 1][j[k] - 1]});
            }
            if (j[k] == m - 1)
            {
                j[k] = 1;
                i[k] += WIDTH;
            }
            else
            {
                j[k]++;
            }
        }
    }

    // Prefix section
    prefix_sequential(X, row_end + 1, n, n, Y, m, D);
}

__global__ void dtw_hybrid_diag_kernel(const double *X, const int n, const double *Y, const int m, double *D)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = 1;
    const int row_end = WIDTH * ((n - 1) / WIDTH);

    int i = 1 + id;
    int j = 1 - id;
    const int diag_num = (m - 1) * ((n - 1) / WIDTH) + WIDTH - 1;
    for (int diag = 0; diag < diag_num; diag++)
    {
        if (row_start <= j && i <= row_end)
        {
            D[i * m + j] = ED(X[i], Y[j]) + min(D[(i - 1) * m + j], min(D[i * m + j - 1], D[(i - 1) * m + j - 1]));
        }
        if (j == m - 1)
        {
            j = 1;
            i += WIDTH;
        }
        else
        {
            j++;
        }
        __syncthreads();
    }
}

void hybrid_parallel(const double *X, const int n, const double *Y, const int m, double **D)
{
    double *h_D;
    double *d_X;
    double *d_Y;
    double *d_D;
    double *d_x;
    double *d_y;

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

    // Diagonal section
    int numBlocks = 1;
    int threadsPerBlock = WIDTH;
    dtw_hybrid_diag_kernel<<<numBlocks, threadsPerBlock>>>(d_X, n, d_Y, m, d_D);
    cudaDeviceSynchronize();

    // Prefix section
    threadsPerBlock = m;
    size_t shmemBytes = m * sizeof(double);
    int start_row = WIDTH * ((n - 1) / WIDTH) + 1;
    dtw_prefix_kernel<<<numBlocks, threadsPerBlock, shmemBytes>>>(d_X, start_row, n, d_Y, m, d_D, d_x, d_y);
    cudaDeviceSynchronize();

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
