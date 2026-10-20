#include <cuda.h>
#include <algorithm>
#include <vector>
#include "prefix.h"

extern double *array_init(const double *, const int, const double *, const int);
extern __global__ void array_init_kernel(const double *, const int, const double *, const int, double *);

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

void prefix_sequential(const double *X, const int row_start, const int row_end, const int n, const double *Y, const int m, double *D)
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
        x[0] = D[i * m + 0];
        for (int j = 1; j < m; j++)
        {
            x[j] = min(D[(i - 1) * m + j], D[(i - 1) * m + j - 1]) - y[j - 1];
        }
        prefix_min(x, m);
        // calculate D
        for (int j = 1; j < m; j++)
        {
            D[i * m + j] = x[j] + y[j];
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

double *prefix_parallel(const double *X, const int row_start, const int row_end, const int n, const double *Y, const int m)
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

    int block_num = 1;
    int thread_num = m;
    size_t shm_byte = m * sizeof(double);
    dtw_prefix_kernel<<<block_num, thread_num, shm_byte>>>(d_X, row_start, row_end, d_Y, m, d_D, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(h_D, d_D, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
    cudaFree(d_x);
    cudaFree(d_y);
    return h_D;
}

std::vector<float> prefix_parallel_benchmark(const double *X, const int row_start, const int row_end, const int n, const double *Y, const int m)
{
    int rounds = 1000;
    int warmup = 10;
    int kernel_num = 1;
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

double *prefix_sequential(const double *X, const int n, const double *Y, const int m)
{
    double *D = array_init(X, n, Y, m);
    prefix_sequential(X, 1, n, n, Y, m, D);
    return D;
}

double *prefix_parallel(const double *X, const int n, const double *Y, const int m)
{
    return prefix_parallel(X, 1, n, n, Y, m);
}

std::vector<float> prefix_parallel_benchmark(const double *X, const int n, const double *Y, const int m)
{
    return prefix_parallel_benchmark(X, 1, n, n, Y, m);
}

double **prefix_parallel_stream(
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
    int thread_num = sequence_length;
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
        dtw_prefix_kernel<<<block_num, thread_num, shm_byte, streams[s]>>>(d_X, 1, sequence_length, d_Y, sequence_length, d_D, d_x, d_y);
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
