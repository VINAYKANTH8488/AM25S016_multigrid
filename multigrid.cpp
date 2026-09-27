#include "multigrid.h"

#include <stdexcept>
#include <cmath>

using namespace std;

std::string cycleName(CycleType c)
{
    return (c == CycleType::V) ? "V-cycle" : "W-cycle";
}

//=====================================================
// Full-weighting restriction (standard 9-point stencil,
// weights 1/16 * [1 2 1; 2 4 2; 1 2 1]).
//
// Because both grids use the boundary-included, n=2^k+1
// convention, the coarse node ic,jc sits at EXACTLY the
// same physical point as fine node (2*ic, 2*jc) -- no
// index offset or clamping is needed.
//=====================================================
vector<double> restrictFullWeighting(
    const vector<double>& fine,
    int nFine,
    int nCoarse)
{
    vector<double> coarse(nCoarse * nCoarse, 0.0);

    for (int jc = 1; jc < nCoarse - 1; jc++)
    {
        for (int ic = 1; ic < nCoarse - 1; ic++)
        {
            int i = 2 * ic;
            int j = 2 * jc;

            double c  = fine[sidx(i,     j,     nFine)];
            double w  = fine[sidx(i - 1, j,     nFine)];
            double e  = fine[sidx(i + 1, j,     nFine)];
            double s  = fine[sidx(i,     j - 1, nFine)];
            double nn = fine[sidx(i,     j + 1, nFine)];
            double sw = fine[sidx(i - 1, j - 1, nFine)];
            double se = fine[sidx(i + 1, j - 1, nFine)];
            double nw = fine[sidx(i - 1, j + 1, nFine)];
            double ne = fine[sidx(i + 1, j + 1, nFine)];

            coarse[sidx(ic, jc, nCoarse)] =
                (sw + 2*s + se + 2*w + 4*c + 2*e + nw + 2*nn + ne) / 16.0;
        }
    }

    return coarse;
}

//=====================================================
// Bilinear prolongation. Every fine node belongs to
// exactly one of four parity classes (even/even,
// odd/even, even/odd, odd/odd in local fine-index
// terms), so each is written exactly once -- no
// double-counting and no nodes left at zero.
//=====================================================
vector<double> prolongBilinear(
    const vector<double>& coarse,
    int nCoarse,
    int nFine)
{
    vector<double> fine(nFine * nFine, 0.0);

    for (int jc = 0; jc < nCoarse; jc++)
    {
        for (int ic = 0; ic < nCoarse; ic++)
        {
            double v = coarse[sidx(ic, jc, nCoarse)];

            int i = 2 * ic;
            int j = 2 * jc;

            // Coincident point
            fine[sidx(i, j, nFine)] = v;

            // East midpoint
            if (ic < nCoarse - 1)
            {
                double vE = coarse[sidx(ic + 1, jc, nCoarse)];
                fine[sidx(i + 1, j, nFine)] = 0.5 * (v + vE);
            }

            // North midpoint
            if (jc < nCoarse - 1)
            {
                double vN = coarse[sidx(ic, jc + 1, nCoarse)];
                fine[sidx(i, j + 1, nFine)] = 0.5 * (v + vN);
            }

            // Diagonal (north-east) midpoint
            if (ic < nCoarse - 1 && jc < nCoarse - 1)
            {
                double vE  = coarse[sidx(ic + 1, jc,     nCoarse)];
                double vN  = coarse[sidx(ic,     jc + 1, nCoarse)];
                double vNE = coarse[sidx(ic + 1, jc + 1, nCoarse)];
                fine[sidx(i + 1, j + 1, nFine)] = 0.25 * (v + vE + vN + vNE);
            }
        }
    }

    return fine;
}

//=====================================================
// MultigridSolver
//=====================================================
MultigridSolver::MultigridSolver(int n0, MGParams params)
    : params_(params)
{
    if (n0 < 3 || ((n0 - 1) & (n0 - 2)) != 0)
    {
        // (n0-1) must be a power of two, i.e. n0 = 2^k + 1
        throw std::invalid_argument(
            "MultigridSolver: finest grid size n0 must be 2^k + 1 "
            "(e.g. 9, 17, 33, 65, 129, 257).");
    }

    int n = n0;
    double h = 1.0 / (n - 1);

    while (true)
    {
        levels_.push_back({n, h});

        bool reachedRequestedDepth =
            (params_.numLevels > 0) &&
            ((int)levels_.size() >= params_.numLevels);

        if (n == 3 || reachedRequestedDepth)
            break;

        n = (n - 1) / 2 + 1;
        h = 1.0 / (n - 1);
    }
}

void MultigridSolver::mgRecurse(
    int lvl,
    vector<double>& u,
    const vector<double>& f)
{
    const MGLevel& L = levels_[lvl];
    int n = L.n;
    double h = L.h;

    bool isCoarsest = (lvl == (int)levels_.size() - 1);

    if (isCoarsest)
    {
        if (n == 3)
        {
            // Exact algebraic solve: single interior unknown,
            // all four neighbours are Dirichlet-zero boundary.
            int p = sidx(1, 1, 3);
            u[p] = f[p] * h * h / 4.0;
        }
        else
        {
            // Hierarchy was truncated above n=3 by the user's
            // requested numLevels: approximate the coarse solve
            // with enough Gauss-Seidel sweeps that it is, for
            // practical purposes, exact.
            smooth(SmootherType::GAUSS_SEIDEL, u, f, n, h, 500, 1.0);
        }
        return;
    }

    // Pre-smoothing
    smooth(params_.smoother, u, f, n, h, params_.preSweeps, params_.omega);

    // Residual and restriction
    vector<double> r = residualStencil(u, f, n, h);

    int nc = levels_[lvl + 1].n;
    vector<double> rc = restrictFullWeighting(r, n, nc);

    // Coarse-grid correction (recursive)
    vector<double> ec(nc * nc, 0.0);

    int gamma = (params_.cycleType == CycleType::W) ? 2 : 1;
    for (int g = 0; g < gamma; g++)
        mgRecurse(lvl + 1, ec, rc);

    // Prolongate and correct
    vector<double> ef = prolongBilinear(ec, nc, n);
    for (int idx = 0; idx < n * n; idx++)
        u[idx] += ef[idx];

    // Post-smoothing
    smooth(params_.smoother, u, f, n, h, params_.postSweeps, params_.omega);
}

void MultigridSolver::cycle(vector<double>& u, const vector<double>& f)
{
    mgRecurse(0, u, f);
}

vector<double> MultigridSolver::solve(
    vector<double>& u,
    const vector<double>& f,
    double tol,
    int maxCycles,
    int& cyclesUsed)
{
    vector<double> history;

    int n = levels_.front().n;
    double h = levels_.front().h;

    history.push_back(l2NormInterior(residualStencil(u, f, n, h), n));

    cyclesUsed = 0;
    for (int c = 1; c <= maxCycles; c++)
    {
        cycle(u, f);

        double res = l2NormInterior(residualStencil(u, f, n, h), n);
        history.push_back(res);
        cyclesUsed = c;

        if (res < tol)
            break;
    }

    return history;
}
