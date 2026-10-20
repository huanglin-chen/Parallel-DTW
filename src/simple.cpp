#include <algorithm>
#include "simple.h"

extern double *array_init(const double *, const int, const double *, const int);

double *simple_sequential(const double *X, const int n, const double *Y, const int m)
{
    double *D = array_init(X, n, Y, m);
    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < m; j++)
        {
            D[i * m + j] = ED(X[i], Y[j]) + std::min(D[(i - 1) * m + j], std::min(D[i * m + j - 1], D[(i - 1) * m + j - 1]));
        }
    }
    return D;
}
