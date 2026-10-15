#include <algorithm>
#include "simple.h"

extern double **array_init_2d(const double *, const int, const double *, const int);

double **simple_sequential(const double *X, const int n, const double *Y, const int m)
{
    double **D = array_init_2d(X, n, Y, m);
    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < m; j++)
        {
            D[i][j] = ED(X[i], Y[j]) + std::min(D[i - 1][j], std::min(D[i][j - 1], D[i - 1][j - 1]));
        }
    }
    return D;
}
