#include "stencil.h"

#include <cmath>

using namespace std;

std::string smootherName(SmootherType s)
{
    switch (s)
    {
        case SmootherType::JACOBI:          return "Jacobi";
        case SmootherType::WEIGHTED_JACOBI: return "Weighted-Jacobi";
        case SmootherType::GAUSS_SEIDEL:    return "Gauss-Seidel";
        case SmootherType::SOR:             return "SOR";
    }
    return "Unknown";
}

//-----------------------------------------------------
// L(u)_{i,j} = ( 4 u_ij - u_{i-1,j} - u_{i+1,j}
//                       - u_{i,j-1} - u_{i,j+1} ) / h^2
// This is the standard 5-point discretization of
// -Laplacian(u), positive definite, matching the sign
// convention rhs(x,y) = +2 pi^2 sin(pi x) sin(pi y).
//-----------------------------------------------------
vector<double> applyOperator(
    const vector<double>& u,
    int n,
    double h)
{
    vector<double> Lu(n * n, 0.0);
    double ih2 = 1.0 / (h * h);

    for (int j = 1; j < n - 1; j++)
    {
        for (int i = 1; i < n - 1; i++)
        {
            int p = sidx(i, j, n);

            double center = u[p];
            double west   = u[sidx(i - 1, j, n)];
            double east   = u[sidx(i + 1, j, n)];
            double south  = u[sidx(i, j - 1, n)];
            double north  = u[sidx(i, j + 1, n)];

            Lu[p] = (4.0 * center - west - east - south - north) * ih2;
        }
    }

    return Lu;
}

vector<double> residualStencil(
    const vector<double>& u,
    const vector<double>& f,
    int n,
    double h)
{
    vector<double> Lu = applyOperator(u, n, h);
    vector<double> r(n * n, 0.0);

    for (int j = 1; j < n - 1; j++)
        for (int i = 1; i < n - 1; i++)
        {
            int p = sidx(i, j, n);
            r[p] = f[p] - Lu[p];
        }

    return r;
}

double l2NormInterior(const vector<double>& v, int n)
{
    double sum = 0.0;

    for (int j = 1; j < n - 1; j++)
        for (int i = 1; i < n - 1; i++)
        {
            double val = v[sidx(i, j, n)];
            sum += val * val;
        }

    return sqrt(sum);
}

void smooth(
    SmootherType type,
    vector<double>& u,
    const vector<double>& f,
    int n,
    double h,
    int sweeps,
    double omega)
{
    double h2 = h * h;
    double diag = 4.0 / h2;

    if (type == SmootherType::JACOBI || type == SmootherType::WEIGHTED_JACOBI)
    {
        double w = (type == SmootherType::JACOBI) ? 1.0 : omega;
        vector<double> unew = u;

        for (int s = 0; s < sweeps; s++)
        {
            for (int j = 1; j < n - 1; j++)
            {
                for (int i = 1; i < n - 1; i++)
                {
                    int p = sidx(i, j, n);

                    double sigma =
                        (u[sidx(i - 1, j, n)] + u[sidx(i + 1, j, n)] +
                         u[sidx(i, j - 1, n)] + u[sidx(i, j + 1, n)]) / h2;

                    double xgs = (f[p] + sigma) / diag;

                    unew[p] = (1.0 - w) * u[p] + w * xgs;
                }
            }
            // Boundary entries of both buffers are never touched by the
            // loop above and stay at 0, so a plain swap (no extra copy)
            // is safe and keeps every interior entry freshly overwritten
            // next sweep.
            u.swap(unew);
        }
        return;
    }

    if (type == SmootherType::GAUSS_SEIDEL || type == SmootherType::SOR)
    {
        double w = (type == SmootherType::GAUSS_SEIDEL) ? 1.0 : omega;

        for (int s = 0; s < sweeps; s++)
        {
            for (int j = 1; j < n - 1; j++)
            {
                for (int i = 1; i < n - 1; i++)
                {
                    int p = sidx(i, j, n);

                    double sigma =
                        (u[sidx(i - 1, j, n)] + u[sidx(i + 1, j, n)] +
                         u[sidx(i, j - 1, n)] + u[sidx(i, j + 1, n)]) / h2;

                    double xgs = (f[p] + sigma) / diag;

                    u[p] = (1.0 - w) * u[p] + w * xgs;
                }
            }
        }
        return;
    }
}
