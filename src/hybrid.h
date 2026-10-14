#ifndef HYBRID_H
#define HYBRID_H

#define ED(a, b) ((a) - (b)) * ((a) - (b))

// The width of the diagonal section.
// Also the number of threads calculating the diagonal section in the hybrid algorithm
#define WIDTH 512

void hybrid_sequential(const double *, const int, const double *, const int, double **);
void hybrid_parallel(const double *, const int, const double *, const int, double **);

#endif