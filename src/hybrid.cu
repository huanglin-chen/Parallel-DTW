#include <cuda.h>
#include <algorithm>
#include <vector>
#include "hybrid.h"

extern void prefix_sequential(const double *, const int, const int, const int, const double *, const int, double *);
extern __global__ void dtw_prefix_kernel(const double *, const int, const int, const double *, const int, double *, double *, double *);
extern __global__ void array_init_kernel(const double *, const int, const double *, const int, double *);
extern double *array_init(const double *, const int, const double *, const int);

double *hybrid_sequential(const double *X, const int n, const double *Y, const int m)
{
    double *D = array_init(X, n, Y, m);
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
                D[i[k] * m + j[k]] = ED(X[i[k]], Y[j[k]]) + std::min(D[(i[k] - 1) * m + j[k]], std::min(D[i[k] * m + j[k] - 1], D[(i[k] - 1) * m + j[k] - 1]));
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

double *hybrid_parallel(const double *X, const int n, const double *Y, const int m)
{
    double *h_D = array_init(X, n, Y, m);
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
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    return h_D;
}

std::vector<float> hybrid_parallel_benchmark(const double *X, const int n, const double *Y, const int m)
{
    int rounds = 1000;
    int warmup = 10;
    int kernel_num = 2;
    std::vector<float> total_ms(kernel_num, 0);

    double *h_D = array_init(X, n, Y, m);
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

double **hybrid_parallel_stream(
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
    // Allocate d_D, d_x, d_y for each stream
    double **d_Ds = new double *[stream_num];
    double **d_xs = new double *[stream_num];
    double **d_ys = new double *[stream_num];
    for (int s = 0; s < stream_num; s++)
    {
        cudaMalloc((void **)&d_Ds[s], sequence_length * sequence_length * sizeof(double));
        cudaMalloc((void **)&d_xs[s], sequence_length * sizeof(double));
        cudaMalloc((void **)&d_ys[s], sequence_length * sizeof(double));
    }
    // Allocate h_D for each pair on pinned memory
    double **h_Ds = new double *[pair_num];
    for (int i = 0; i < pair_num; i++)
    {
        h_Ds[i] = new double[sequence_length * sequence_length];
        cudaHostRegister(h_Ds[i], sequence_length * sequence_length * sizeof(double), cudaHostRegisterDefault);
    }

    // Create cuda streams
    std::vector<cudaStream_t> streams(stream_num);
    for (int s = 0; s < stream_num; s++)
    {
        cudaStreamCreate(&streams[s]);
    }

    // Launch kernels
    int block_num = 1;
    int thread_num = WIDTH;
    size_t shm_byte = sequence_length * sizeof(double);
    for(int i = 0; i < pair_num; i++)
    {
        int s = i % stream_num;
        double *d_X = &d_sequences[pairs[i].first * sequence_length];
        double *d_Y = &d_sequences[pairs[i].second * sequence_length];
        double *d_x = d_xs[s];
        double *d_y = d_ys[s];
        double *h_D = h_Ds[i];
        double *d_D = d_Ds[s];
        array_init_kernel<<<1, 2, 0, streams[s]>>>(d_X, sequence_length, d_Y, sequence_length, d_D);
        dtw_hybrid_diag_kernel<<<block_num, thread_num, 0, streams[s]>>>(d_X, sequence_length, d_Y, sequence_length, d_D);
        
        thread_num = sequence_length;
        int row_start = WIDTH * ((sequence_length - 1) / WIDTH) + 1;
        dtw_prefix_kernel<<<block_num, thread_num, shm_byte, streams[s]>>>(d_X, row_start, sequence_length, d_Y, sequence_length, d_D, d_x, d_y);
        
        cudaMemcpyAsync(h_D, d_D, sequence_length * sequence_length * sizeof(double), cudaMemcpyDeviceToHost, streams[s]);
    }
    for (int s = 0; s < stream_num; s++)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Destroy streams
    for (int s = 0; s < stream_num; s++)
    {
        cudaStreamDestroy(streams[s]);
    }

    cudaFree(d_sequences);
    for (int s = 0; s < stream_num; s++)
    {
        cudaFree(d_Ds[s]);
        cudaFree(d_xs[s]);
        cudaFree(d_ys[s]);
    }
    delete[] d_Ds;
    delete[] d_xs;
    delete[] d_ys;
    for (int i = 0; i < pair_num; i++)
    {
        cudaHostUnregister(h_Ds[i]);
    }
    return h_Ds;
}
