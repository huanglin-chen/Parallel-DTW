#ifndef HYBRID_H
#define HYBRID_H

#define ED(a, b) ((a) - (b)) * ((a) - (b))

// The width of the diagonal section.
// Also the number of threads calculating the diagonal section in the hybrid algorithm
#define WIDTH 256

double **hybrid_sequential(const double *, const int, const double *, const int);
double **hybrid_parallel(const double *, const int, const double *, const int);

std::vector<float> hybrid_parallel_benchmark(const double *, const int, const double *, const int);

#endif