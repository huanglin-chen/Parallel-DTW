#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include <chrono>

#include "simple.h"
#include "diagonal.h"
#include "prefix.h"
#include "hybrid.h"

#define RED "\033[1;31m"
#define GREEN "\033[1;32m"
#define RESET "\033[0m"

typedef void (*dtw_algorithm)(const double *, const int, const double *, const int, double **);

static double *read_sequence(int &sequence_num, int &sequence_length)
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

static void verify(int rounds, std::string algorithm_name, dtw_algorithm algorithm, double *sequences, const int sequence_num, const int sequence_length)
{
    bool pass = true;
    int n = sequence_length;
    int m = sequence_length;

    double **D1 = new double *[n];
    double **D2 = new double *[n];
    for (int i = 0; i < n; i++)
    {
        D1[i] = new double[m];
        D2[i] = new double[m];
    }

    for (int round = 0; round < rounds; round++)
    {
        double *X = &sequences[(round % sequence_num) * sequence_length];
        double *Y = &sequences[((round + 1) % sequence_num) * sequence_length];
        // Initialize the cost matrix
        D1[0][0] = ED(X[0], Y[0]);
        for (int i = 1; i < n; i++)
        {
            D1[i][0] = D1[i - 1][0] + ED(X[i], Y[0]);
        }
        for (int j = 1; j < m; j++)
        {
            D1[0][j] = D1[0][j - 1] + ED(X[0], Y[j]);
        }
        for (int i = 1; i < n; i++)
        {
            for (int j = 1; j < m; j++)
            {
                D1[i][j] = std::numeric_limits<double>::max();
            }
        }
        for (int i = 0; i < n; i++)
        {
            std::copy(D1[i], D1[i] + m, D2[i]);
        }

        // Compare the algorithm with simple DTW algorithm
        simple_sequential(X, n, Y, m, D1);
        algorithm(X, n, Y, m, D2);

        // verify the result
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (std::abs(D1[i][j] - D2[i][j]) > 1e-4)
                {
#ifdef DEBUG
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
#endif
                    pass = false;
                    goto cleanup;
                }
            }
        }
    }

cleanup:
    for (int i = 0; i < n; i++)
    {
        delete[] D1[i];
        delete[] D2[i];
    }
    delete[] D1;
    delete[] D2;

    if (pass)
    {
        std::cout << GREEN << "[ PASS ] " << algorithm_name << RESET << std::endl;
    }
    else
    {
        std::cout << RED << "[ FAIL ] " << algorithm_name << RESET << std::endl;
    }
}

static double cost_matrix_benchmark(const int rounds, dtw_algorithm algorithm, double *sequences, const int sequence_num, const int sequence_length)
{
    if (rounds <= 0)
    {
        return 0;
    }
    double ret = 0;
    int n = sequence_length;
    int m = sequence_length;

    double **D = new double *[n];
    for (int i = 0; i < n; i++)
    {
        D[i] = new double[m];
    }

    int warmup = rounds / 10;
    for (int round = 0; round < rounds; round++)
    {
        double *X = &sequences[(round % sequence_num) * sequence_length];
        double *Y = &sequences[((round + 1) % sequence_num) * sequence_length];
        // Initialize the cost matrix
        D[0][0] = ED(X[0], Y[0]);
        for (int i = 1; i < n; i++)
        {
            D[i][0] = D[i - 1][0] + ED(X[i], Y[0]);
        }
        for (int j = 1; j < m; j++)
        {
            D[0][j] = D[0][j - 1] + ED(X[0], Y[j]);
        }
        for (int i = 1; i < n; i++)
        {
            for (int j = 1; j < m; j++)
            {
                D[i][j] = std::numeric_limits<double>::max();
            }
        }

        auto start = std::chrono::steady_clock::now();

        algorithm(X, n, Y, m, D);

        auto end = std::chrono::steady_clock::now();
        if (round >= warmup)
            ret += std::chrono::duration<double, std::milli>(end - start).count();
    }

    for (int i = 0; i < n; i++)
    {
        delete[] D[i];
    }
    delete[] D;

    return ret / (rounds - warmup);
}

int main()
{
    int sequence_num = 0;
    int sequence_length = 0;
    double *sequences = read_sequence(sequence_num, sequence_length);

    int rounds = 1000;
    verify(rounds, "diagonal_sequential", diagonal_sequential, sequences, sequence_num, sequence_length);
    verify(rounds, "diagonal_parallel", diagonal_parallel, sequences, sequence_num, sequence_length);
    verify(rounds, "prefix_sequential", prefix_sequential, sequences, sequence_num, sequence_length);
    verify(rounds, "prefix_parallel", prefix_parallel, sequences, sequence_num, sequence_length);
    verify(rounds, "hybrid_sequential", hybrid_sequential, sequences, sequence_num, sequence_length);
    verify(rounds, "hybrid_parallel", hybrid_parallel, sequences, sequence_num, sequence_length);

    delete[] sequences;
    return 0;
}