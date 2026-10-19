#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <vector>
#include <random>
#ifdef DEBUG
#include <iomanip>
#endif

#include "simple.h"
#include "diagonal.h"
#include "prefix.h"
#include "hybrid.h"

#define RED "\033[1;31m"
#define GREEN "\033[1;32m"
#define RESET "\033[0m"

typedef double **(*dtw_algorithm)(const double *, const int, const double *, const int);
typedef std::vector<float> (*dtw_algorithm_benchmark)(const double *, const int, const double *, const int);
typedef std::vector<double **> (*dtw_algorithm_stream)(std::vector<std::pair<int, int>> &, double *, int , int, int);

double *array_init_1d(const double *X, const int n, const double *Y, const int m)
{
    double *h_D = new double[n * m];
    h_D[0 * m + 0] = ED(X[0], Y[0]);
    for (int i = 1; i < n; i++)
    {
        h_D[i * m + 0] = ED(X[i], Y[0]) + h_D[(i - 1) * m + 0];
    }
    for (int j = 1; j < m; j++)
    {
        h_D[0 * m + j] = ED(X[0], Y[j]) + h_D[0 * m + j - 1];
    }
    return h_D;
}
double **array_init_2d(const double *X, const int n, const double *Y, const int m)
{
    double **D = new double *[n];
    for (int i = 0; i < n; i++)
    {
        D[i] = new double[m];
    }
    D[0][0] = ED(X[0], Y[0]);
    for (int i = 1; i < n; i++)
    {
        D[i][0] = D[i - 1][0] + ED(X[i], Y[0]);
    }
    for (int j = 1; j < m; j++)
    {
        D[0][j] = D[0][j - 1] + ED(X[0], Y[j]);
    }
    return D;
}

double *read_sequence(int &sequence_num, int &sequence_length)
{
    std::ifstream dataset("UCR_TS_Archive_2015/StarLightCurves/StarLightCurves_TRAIN");
    if (!dataset.is_open())
    {
        std::cerr << "Can not open the dataset file" << std::endl;
        exit(-1);
    }

    std::string line;

    sequence_num = 0;
    sequence_length = 0;
    while (std::getline(dataset, line))
    {
        if (sequence_length == 0)
        {
            sequence_length = std::count(line.begin(), line.end(), ',');
        }
        sequence_num++;
    }
    dataset.clear(); // Clear EOF flag
    dataset.seekg(0, std::ios::beg);

    double *sequences = new double[sequence_num * sequence_length];
    int current = 0;
    while (std::getline(dataset, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ','); // Read the id
        while (std::getline(ss, token, ','))
        {
            sequences[current++] = std::stod(token);
        }
    }
    return sequences;
}

void verify(std::string algorithm_name, dtw_algorithm algorithm, double *sequences, const int sequence_num, const int sequence_length)
{
    int rounds = 1000;
    bool pass = true;
    int n = sequence_length;
    int m = sequence_length;

#ifdef DEBUG
    double max_e = 0;
#endif
    for (int round = 0; round < rounds && pass; round++)
    {
        double *X = &sequences[(round % sequence_num) * sequence_length];
        double *Y = &sequences[((round + 1) % sequence_num) * sequence_length];

        // Compare the algorithm with simple DTW algorithm
        double **D1 = simple_sequential(X, n, Y, m);
        double **D2 = algorithm(X, n, Y, m);

        // verify the result
        for (int i = 0; i < n && pass; i++)
        {
            for (int j = 0; j < m && pass; j++)
            {
                pass = pass && (std::abs(D1[i][j] - D2[i][j]) <= 1e-4);
#ifdef DEBUG
                max_e = std::max(max_e, std::abs(D1[i][j] - D2[i][j]));
                if (!pass)
                {
                    std::cout << "Failed at: [" << i << ", " << j << "]" << std::endl;
                    for (int l = 0; l < 10; l++)
                    {
                        std::cout << D1[i][l] << " ";
                    }
                    std::cout << std::endl;
                    for (int l = 0; l < 10; l++)
                    {
                        std::cout << D2[i][l] << " ";
                    }
                    std::cout << std::endl;
                }
#endif
            }
        }
        for (int i = 0; i < n; i++)
        {
            delete[] D1[i];
            delete[] D2[i];
        }
        delete[] D1;
        delete[] D2;
    }
    if (pass)
    {
#ifdef DEBUG
        std::cout << GREEN << "[ PASS ] " << algorithm_name << RESET << " (error <= " << max_e << ")" << std::endl;
#else
        std::cout << GREEN << "[ PASS ] " << algorithm_name << RESET << std::endl;
#endif
    }
    else
    {
        std::cout << RED << "[ FAIL ] " << algorithm_name << RESET << std::endl;
    }
}

void cost_matrix_benchmark(std::string algorithm_name, dtw_algorithm algorithm, double *sequences, const int sequence_num, const int sequence_length)
{
    int rounds = 1000;
    int warmup = 10;
    double total_ms = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, sequence_num - 1);
    int n = sequence_length;
    int m = sequence_length;

    for (int round = 0; round < warmup; round++)
    {
        double *X = &sequences[dist(gen) * sequence_length];
        double *Y = &sequences[dist(gen) * sequence_length];
        double **D = algorithm(X, n, Y, m);
        for (int i = 0; i < n; i++)
        {
            delete[] D[i];
        }
        delete[] D;
    }
    for (int round = 0; round < rounds; round++)
    {
        double *X = &sequences[dist(gen) * sequence_length];
        double *Y = &sequences[dist(gen) * sequence_length];
        auto start = std::chrono::steady_clock::now();

        double **D = algorithm(X, n, Y, m);

        auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();

        for (int i = 0; i < n; i++)
        {
            delete[] D[i];
        }
        delete[] D;
    }

    std::cout << "[ Cost Matrix ] " << algorithm_name << ": \t" << total_ms / rounds << " ms." << std::endl;
}

void kernel_benchmark(std::string algorithm_name, dtw_algorithm_benchmark algorithm, double *sequences, const int sequence_num, const int sequence_length)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, sequence_num - 1);
    int n = sequence_length;
    int m = sequence_length;
    double *X = &sequences[dist(gen) * sequence_length];
    double *Y = &sequences[dist(gen) * sequence_length];
    std::vector<float> total_ms = algorithm(X, n, Y, m);

    std::cout << "[ Kernel ] " << algorithm_name << ": \t";
    for (float ms : total_ms)
    {
        std::cout << ms << " ";
    }
    std::cout << "ms." << std::endl;
}

void stream_benchmark(std::string algorithm_name, dtw_algorithm_stream algorithm, double *sequences, const int sequence_num, const int sequence_length)
{
    int stream_num = 16;
    int rounds = 100;
    int warmup = 10;

    std::vector<std::pair<int, int>> pairs;
    for(int round = 0; round < warmup; round++)
    {
        pairs.push_back({round % sequence_num, (round + 1) % sequence_num});
    }

    std::vector<double **> Ds = algorithm(pairs, sequences, sequence_num, sequence_length, stream_num);
    for(int i = 0; i < warmup; i++)
    {
        for (int j = 0; j < sequence_length; j++)
        {
            delete[] Ds[i][j];
        }
        delete[] Ds[i];
    }

    for(int round = warmup; round < rounds; round++)
    {
        pairs.push_back({round % sequence_num, (round + 1) % sequence_num});
    }

    auto start = std::chrono::steady_clock::now();
    
    Ds = algorithm(pairs, sequences, sequence_num, sequence_length, stream_num);

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    for(int i = 0; i < rounds; i++)
    {
        for (int j = 0; j < sequence_length; j++)
        {
            delete[] Ds[i][j];
        }
        delete[] Ds[i];
    }

    std::cout << "[ Stream ] " << algorithm_name << ": \t" << ms / rounds << " ms for " << stream_num << " streams." << std::endl;
}

int main()
{
    int sequence_num = 0;
    int sequence_length = 0;
    double *sequences = read_sequence(sequence_num, sequence_length);

    std::vector<std::pair<std::string, dtw_algorithm>> test_case1{
        {"simple_sequential", simple_sequential},
        {"diagonal_sequential", diagonal_sequential},
        {"diagonal_parallel", diagonal_parallel},
        {"prefix_sequential", prefix_sequential},
        {"prefix_parallel", prefix_parallel},
        {"hybrid_sequential", hybrid_sequential},
        {"hybrid_parallel", hybrid_parallel}};
    std::vector<std::pair<std::string, dtw_algorithm_benchmark>> test_case2{
        {"diagonal_parallel", diagonal_parallel_benchmark},
        {"prefix_parallel", prefix_parallel_benchmark},
        {"hybrid_parallel", hybrid_parallel_benchmark}};

    for (int i = 1; i < 7; i++)
    {
        verify(test_case1[i].first, test_case1[i].second, sequences, sequence_num, sequence_length);
    }
    for (int i = 0; i < 7; i++)
    {
        cost_matrix_benchmark(test_case1[i].first, test_case1[i].second, sequences, sequence_num, sequence_length);
    }
    for (int i = 0; i < 3; i++)
    {
        kernel_benchmark(test_case2[i].first, test_case2[i].second, sequences, sequence_num, sequence_length);
    }

    delete[] sequences;
    return 0;
}