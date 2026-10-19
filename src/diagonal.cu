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

__global__ void array_init(const double *X, const int n, const double *Y, const int m, double *D)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id == 0)
    {
        D[0 * m + 0] = ED(X[0], Y[0]);
    }
    __syncthreads();

    if(id == 0)
    {
        for (int i = 1; i < n; i++)
        {
            D[i * m + 0] = ED(X[i], Y[0]) + D[(i - 1) * m + 0];
        }
    }
    else if(id == 1)
    {
        for (int j = 1; j < m; j++)
        {
            D[0 * m + j] = ED(X[0], Y[j]) + D[0 * m + j - 1];
        }
    }
}


std::vector<double **> diagonal_parallel_stream(
    std::vector<std::pair<int, int>> &pairs,
    double *sequences, 
    int sequence_num, 
    int sequence_length, 
    int stream_num)
{
    int pair_num = pairs.size();

    // Copy the whole dataset to gpu
    double *d_sequences;
    cudaMalloc((void **)&d_sequences, sequence_num * sequence_length * sizeof(double));
    cudaMemcpy(d_sequences, sequences, sequence_num * sequence_length * sizeof(double), cudaMemcpyHostToDevice);
    // Allocate d_D for each stream
    std::vector<double *> d_Ds(stream_num, nullptr);
    for(int s = 0; s < stream_num; s++)
    {
        cudaMalloc((void **)&d_Ds[s], sequence_length * sequence_length * sizeof(double));
    }
    // Allocate h_Ds for each pair on pinned memory
    std::vector<double *> h_Ds(pair_num, nullptr);
    for(int i = 0; i < pair_num; i++)
    {
        h_Ds[i] = new double[sequence_length * sequence_length];
    }

    int block_num = 1;
    int thread_num = 1024;
    std::vector<cudaStream_t> streams(stream_num);
    for(int s = 0; s < stream_num; s++)
    {
        cudaStreamCreate(&streams[s]);
    }
    for(int i = 0; i < pair_num; i++)
    {
        int s = i % stream_num;
        double *d_X = &d_sequences[pairs[i].first * sequence_length];
        double *d_Y = &d_sequences[pairs[i].second * sequence_length];
        array_init<<<1, 2, 0, streams[s]>>>(d_X, sequence_length, d_Y, sequence_length, d_Ds[s]);
        dtw_diag_kernel<<<block_num, thread_num, 0, streams[s]>>>(d_X, sequence_length, d_Y, sequence_length, d_Ds[s]);
        cudaMemcpyAsync(h_Ds[i], d_Ds[s], sequence_length * sequence_length * sizeof(double), cudaMemcpyDeviceToHost, streams[s]);
    }
    for(int s = 0; s < stream_num; s++)
    {
        cudaStreamSynchronize(streams[s]);
    }
    for(int s = 0; s < stream_num; s++)
    {
        cudaStreamDestroy(streams[s]);
    }

    std::vector<double **> Ds(pair_num, nullptr);
    for(int i = 0; i < pair_num; i++)
    {
        Ds[i] = new double *[sequence_length];
        for (int j = 0; j < sequence_length; j++)
        {
            Ds[i][j] = new double[sequence_length];
            memcpy(Ds[i][j], &h_Ds[i][j * sequence_length], sequence_length * sizeof(double));
        }
    }

    for(int i = 0; i < pair_num; i++)
    {
        delete[] h_Ds[i];
    }
    for(int s = 0; s < stream_num; s++)
    {
        cudaFree(d_Ds[s]);
    }
    cudaFree(d_sequences);

    return Ds;
}
