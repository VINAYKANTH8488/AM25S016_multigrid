#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <vector>
#include <string>
#include "stencil.h"

enum class CycleType { V, W };

std::string cycleName(CycleType c);

//=====================================================
// One entry in the grid hierarchy.
//=====================================================
struct MGLevel
{
    int n;       // nodes per side, including boundary
    double h;    // spacing = 1/(n-1)
};

//=====================================================
// User-tunable parameters for a multigrid solve. This
// bundles everything the assignment asks to be varied
// and compared: smoother, relaxation factor, number of
// pre/post smoothing sweeps, cycle type (V or W), and
// how many levels deep the hierarchy goes.
//=====================================================
struct MGParams
{
    SmootherType smoother = SmootherType::GAUSS_SEIDEL;
    double omega          = 1.0;   // used by WEIGHTED_JACOBI / SOR
    int preSweeps         = 2;
    int postSweeps        = 2;
    CycleType cycleType   = CycleType::V;
    int numLevels         = -1;    // -1 = go all the way down to n=3
};

//=====================================================
// Pure geometric restriction / prolongation (also
// exposed standalone so they can be unit-tested, as in
// the original assignment code).
//=====================================================
std::vector<double> restrictFullWeighting(
    const std::vector<double>& fine,
    int nFine,
    int nCoarse);

std::vector<double> prolongBilinear(
    const std::vector<double>& coarse,
    int nCoarse,
    int nFine);

//=====================================================
// MultigridSolver: builds the level hierarchy once for
// a given finest grid size and reuses it across many
// solves/cycles (so batch comparisons don't redo setup
// work every time).
//=====================================================
class MultigridSolver
{
public:
    // n0 must be 2^k + 1 for some k >= 1 (e.g. 17, 33, 65, 129, 257).
    MultigridSolver(int n0, MGParams params);

    // Run exactly one V- or W-cycle in place on (u, f),
    // both sized n0*n0, at the finest level.
    void cycle(std::vector<double>& u, const std::vector<double>& f);

    // Convenience: run cycles until the interior residual
    // L2 norm drops below tol or maxCycles is reached.
    // Returns the residual-norm history (index 0 = initial).
    std::vector<double> solve(
        std::vector<double>& u,
        const std::vector<double>& f,
        double tol,
        int maxCycles,
        int& cyclesUsed);

    int finestN()  const { return levels_.front().n; }
    double finestH() const { return levels_.front().h; }
    int depth()    const { return (int)levels_.size(); }

private:
    void mgRecurse(int lvl, std::vector<double>& u, const std::vector<double>& f);

    std::vector<MGLevel> levels_;
    MGParams params_;
};

#endif
