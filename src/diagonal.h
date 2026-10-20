#ifndef DIAGONAL_H
#define DIAGONAL_H

#define ED(a, b) ((a) - (b)) * ((a) - (b))

double *diagonal_sequential(const double *, const int, const double *, const int);
double *diagonal_parallel(const double *, const int, const double *, const int);

std::vector<float> diagonal_parallel_benchmark(const double *, const int, const double *, const int);
double **diagonal_parallel_stream(std::vector<std::pair<int, int>> &, double *, int, int, int);
#endif