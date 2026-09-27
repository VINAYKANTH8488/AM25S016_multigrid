#ifndef STENCIL_H
#define STENCIL_H

#include <vector>
#include <string>

//=====================================================
// Grid convention (IMPORTANT):
//
// Each level stores an n x n array of nodes INCLUDING
// the Dirichlet boundary (indices i,j = 0 and n-1).
// Interior unknowns are i,j = 1 .. n-2.
// Spacing: h = 1/(n-1).
//
// This "vertex-centered, boundary-included" convention
// is what makes the standard geometric coarsening
//      n_coarse = (n_fine - 1)/2 + 1
// exactly align coarse and fine nodes at the SAME
// physical locations (fine index 2*ic <-> coarse index
// ic). It requires n = 2^k + 1 on the finest grid so
// that every coarsening divides evenly, all the way
// down to the trivial 3x3 (1 interior unknown) grid.
//
// This is the classic convention used in Briggs, Henson
// & McCormick, "A Multigrid Tutorial", 2nd ed., SIAM,
// 2000 -- used here as the reference method.
//=====================================================

enum class SmootherType
{
    JACOBI,
    WEIGHTED_JACOBI,
    GAUSS_SEIDEL,
    SOR
};

std::string smootherName(SmootherType s);

// Flat index helper: row-major, i = x-index, j = y-index
inline int sidx(int i, int j, int n) { return j * n + i; }

// Apply the negative-Laplacian 5-point operator to u.
// Result is meaningful only at interior nodes; boundary
// entries of the returned vector are left as 0.
std::vector<double> applyOperator(
    const std::vector<double>& u,
    int n,
    double h);

// r = f - L(u), interior nodes only (boundary rows = 0)
std::vector<double> residualStencil(
    const std::vector<double>& u,
    const std::vector<double>& f,
    int n,
    double h);

// L2 norm over interior nodes only
double l2NormInterior(const std::vector<double>& v, int n);

// In-place smoothing sweeps. omega is used only by
// WEIGHTED_JACOBI and SOR (ignored otherwise).
void smooth(
    SmootherType type,
    std::vector<double>& u,
    const std::vector<double>& f,
    int n,
    double h,
    int sweeps,
    double omega);

#endif
