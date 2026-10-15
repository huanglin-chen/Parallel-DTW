#include <cuda.h>
#include <algorithm>
#include <vector>
#include "hybrid.h"

extern void prefix_sequential(const double *, const int, const int, const int, const double *, const int, double **);
extern __global__ void dtw_prefix_kernel(const double *, const int, const int, const double *, const int, double *, double *, double *);
extern double *array_init_1d(const double *, const int, const double *, const int);
extern double **array_init_2d(const double *, const int, const double *, const int);

double **hybrid_sequential(const double *X, const int n, const double *Y, const int m)
{
    double **D = array_init_2d(X, n, Y, m);
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

    return D;
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

double **hybrid_parallel(const double *X, const int n, const double *Y, const int m)
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

    // Diagonal section
    int block_num = 1;
    int thread_num = WIDTH;
    dtw_hybrid_diag_kernel<<<block_num, thread_num>>>(d_X, n, d_Y, m, d_D);

    // Prefix section
    thread_num = m;
    size_t shm_byte = m * sizeof(double);
    int row_start = WIDTH * ((n - 1) / WIDTH) + 1;
    dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, n, d_Y, m, d_D, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(h_D, d_D, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    double **D = new double *[n];
    for (int i = 0; i < n; i++)
    {
        D[i] = new double[m];
        memcpy(D[i], &h_D[i * m], m * sizeof(double));
    }
    delete[] h_D;
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    return D;
}

std::vector<float> hybrid_parallel_benchmark(const double *X, const int n, const double *Y, const int m)
{
    int rounds = 100;
    int warmup = 10;
    int kernel_num = 2;
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
    int thread_num = WIDTH;
    size_t shm_byte = m * sizeof(double);
    int row_start = WIDTH * ((n - 1) / WIDTH) + 1;
    for (int round = 0; round < warmup; round++)
    {
        thread_num = WIDTH;
        dtw_hybrid_diag_kernel<<<block_num, thread_num>>>(d_X, n, d_Y, m, d_D);
        thread_num = m;
        dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, n, d_Y, m, d_D, d_x, d_y);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start1;
    cudaEvent_t start2;
    cudaEvent_t end;
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&end);
    for (int round = 0; round < rounds; round++)
    {
        cudaEventRecord(start1);

        thread_num = WIDTH;
        dtw_hybrid_diag_kernel<<<block_num, thread_num>>>(d_X, n, d_Y, m, d_D);

        cudaEventRecord(start2);
        cudaEventSynchronize(start2);

        thread_num = m;
        dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, n, d_Y, m, d_D, d_x, d_y);

        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start1, start2);
        total_ms[0] += ms;
        cudaEventElapsedTime(&ms, start2, end);
        total_ms[1] += ms;
    }
    for (int i = 0; i < kernel_num; i++)
    {
        total_ms[i] /= rounds;
    }
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(end);

    delete[] h_D;
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    return total_ms;
}