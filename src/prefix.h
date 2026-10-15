#ifndef PREFIX_H
#define PREFIX_H

#define ED(a, b) ((a) - (b)) * ((a) - (b))

double **prefix_sequential(const double *, const int, const double *, const int);
double **prefix_parallel(const double *, const int, const double *, const int);

std::vector<float> prefix_parallel_benchmark(const double *, const int, const double *, const int);

#endif