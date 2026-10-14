#ifndef DIAGONAL_H
#define DIAGONAL_H

#define ED(a, b) ((a) - (b)) * ((a) - (b))

void diagonal_sequential(const double *, const int, const double *, const int, double **);
void diagonal_parallel(const double *, const int, const double *, const int, double **);

#endif