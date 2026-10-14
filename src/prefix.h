#ifndef PREFIX_H
#define PREFIX_H

#define ED(a, b) ((a) - (b)) * ((a) - (b))

void prefix_sequential(const double *, const int, const double *, const int, double **);
void prefix_parallel(const double *, const int, const double *, const int, double **);

#endif