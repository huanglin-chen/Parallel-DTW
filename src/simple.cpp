#include <algorithm>
#include "simple.h"

void simple_sequential(const double *X, const int n, const double *Y, const int m, double **D)
{
    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < m; j++)
        {
            D[i][j] = ED(X[i], Y[j]) + std::min(D[i - 1][j], std::min(D[i][j - 1], D[i - 1][j - 1]));
        }
    }
}
