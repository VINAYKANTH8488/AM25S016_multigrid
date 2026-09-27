#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <numeric>
#include <limits>
#include <string>

#include "poisson.h"
#include "smoothers.h"
#include "stencil.h"
#include "multigrid.h"

using namespace std;
using Clock = chrono::high_resolution_clock;

//=====================================================
// Test problem (same manufactured / analytical solution
// used throughout, on the unit square):
//
//   -Laplacian(u) = 2 pi^2 sin(pi x) sin(pi y)
//   u = 0 on the boundary
//
// Exact solution: u(x,y) = sin(pi x) sin(pi y)
// (already declared in poisson.h / defined in poisson.cpp
//  and reused here unchanged).
//=====================================================

//-----------------------------------------------------
// Build f (RHS) and the exact solution on an n x n
// boundary-included grid (h = 1/(n-1)). Both arrays are
// zero on the boundary, matching the Dirichlet data.
//-----------------------------------------------------
static void buildProblem(
    int n, double h,
    vector<double>& f,
    vector<double>& uExact)
{
    f.assign(n * n, 0.0);
    uExact.assign(n * n, 0.0);

    for (int j = 1; j < n - 1; j++)
    {
        for (int i = 1; i < n - 1; i++)
        {
            double x = i * h;
            double y = j * h;
            int p = sidx(i, j, n);

            f[p]      = rhs(x, y);
            uExact[p] = exact(x, y);
        }
    }
}

static double l2ErrorInterior(
    const vector<double>& u,
    const vector<double>& uExact,
    int n)
{
    double sum = 0.0;
    for (int j = 1; j < n - 1; j++)
        for (int i = 1; i < n - 1; i++)
        {
            double d = u[sidx(i, j, n)] - uExact[sidx(i, j, n)];
            sum += d * d;
        }
    return sqrt(sum);
}

// Geometric-mean convergence factor over the tail of the
// residual history (skips the first cycle, which is often
// an atypically large drop/rise while transients settle).
static double convergenceFactor(const vector<double>& hist)
{
    if (hist.size() < 3) return numeric_limits<double>::quiet_NaN();

    double logSum = 0.0;
    int count = 0;
    for (size_t i = 2; i < hist.size(); i++)
    {
        if (hist[i - 1] > 1e-300 && hist[i] > 0.0)
        {
            logSum += log(hist[i] / hist[i - 1]);
            count++;
        }
    }
    if (count == 0) return numeric_limits<double>::quiet_NaN();
    return exp(logSum / count);
}

static int nFromK(int k) { return (1 << k) + 1; } // n = 2^k + 1

//=====================================================
// [1] Legacy single-grid demo
// Reproduces the Assignment 1-4 dense-matrix solvers
// (Jacobi / Gauss-Seidel / SOR) exactly as before, for
// continuity and as the "no multigrid" baseline.
//=====================================================
static void runLegacyDemo()
{
    int nx, ny;
    cout << "Enter nx : ";
    cin >> nx;
    cout << "Enter ny : ";
    cin >> ny;

    int N = nx * ny;
    double hx = 1.0 / (nx + 1);
    double hy = 1.0 / (ny + 1);

    vector<vector<double>> A = buildPoissonMatrix(nx, ny);

    vector<double> uExact(N), b(N);
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
        {
            double x = (i + 1) * hx;
            double y = (j + 1) * hy;
            int p = idx(i, j, nx);
            uExact[p] = exact(x, y);
            b[p] = rhs(x, y);
        }

    int maxIter;
    cout << "Enter smoother iterations : ";
    cin >> maxIter;

    double omega;
    cout << "Enter SOR omega : ";
    cin >> omega;

    vector<double> uJ(N, 0.0), uGS(N, 0.0), uSOR(N, 0.0);

    auto t0 = Clock::now();
    Jacobi(A, uJ, b, maxIter);
    auto t1 = Clock::now();
    GaussSeidel(A, uGS, b, maxIter);
    auto t2 = Clock::now();
    SOR(A, uSOR, b, maxIter, omega);
    auto t3 = Clock::now();

    auto residNorm = [&](const vector<double>& u)
    {
        double s = 0.0;
        for (int i = 0; i < N; i++)
        {
            double Au = 0.0;
            for (int j = 0; j < N; j++) Au += A[i][j] * u[j];
            double r = b[i] - Au;
            s += r * r;
        }
        return sqrt(s);
    };

    cout << "\nLegacy Solver Comparison (dense matrix, single grid)\n";
    cout << "-----------------------------------------------------\n";
    cout << left << setw(16) << "Solver" << setw(16) << "Residual"
         << setw(14) << "Time (ms)" << "\n";
    cout << left << setw(16) << "Jacobi"
         << setw(16) << residNorm(uJ)
         << setw(14) << chrono::duration<double, milli>(t1 - t0).count() << "\n";
    cout << left << setw(16) << "Gauss-Seidel"
         << setw(16) << residNorm(uGS)
         << setw(14) << chrono::duration<double, milli>(t2 - t1).count() << "\n";
    cout << left << setw(16) << "SOR (w=" + to_string(omega) + ")"
         << setw(16) << residNorm(uSOR)
         << setw(14) << chrono::duration<double, milli>(t3 - t2).count() << "\n";
}

//=====================================================
// [2] Single interactive multigrid solve
//=====================================================
static SmootherType askSmoother()
{
    cout << "Smoother  [1] Jacobi  [2] Weighted-Jacobi  [3] Gauss-Seidel  [4] SOR : ";
    int c; cin >> c;
    switch (c)
    {
        case 1: return SmootherType::JACOBI;
        case 2: return SmootherType::WEIGHTED_JACOBI;
        case 4: return SmootherType::SOR;
        default: return SmootherType::GAUSS_SEIDEL;
    }
}

static void runInteractiveMultigrid()
{
    cout << "Finest grid exponent k (grid = 2^k + 1 nodes/side, e.g. 6 -> 65) : ";
    int k; cin >> k;
    int n0 = nFromK(k);

    cout << "Number of levels (0 = go all the way to the 3x3 coarsest grid) : ";
    int nl; cin >> nl;

    MGParams p;
    p.smoother = askSmoother();

    if (p.smoother == SmootherType::WEIGHTED_JACOBI)
    {
        cout << "Jacobi weight omega (0.6-0.9 typical) : ";
        cin >> p.omega;
    }
    else if (p.smoother == SmootherType::SOR)
    {
        cout << "SOR omega (1 < omega < 2) : ";
        cin >> p.omega;
    }

    cout << "Pre-smoothing sweeps  : "; cin >> p.preSweeps;
    cout << "Post-smoothing sweeps : "; cin >> p.postSweeps;

    cout << "Cycle type [1] V-cycle [2] W-cycle : ";
    int ct; cin >> ct;
    p.cycleType = (ct == 2) ? CycleType::W : CycleType::V;

    p.numLevels = (nl <= 0) ? -1 : nl;

    cout << "Convergence tolerance (residual L2 norm) : ";
    double tol; cin >> tol;

    cout << "Max cycles : ";
    int maxCycles; cin >> maxCycles;

    MultigridSolver solver(n0, p);

    vector<double> f, uExact;
    buildProblem(n0, solver.finestH(), f, uExact);
    vector<double> u(n0 * n0, 0.0);

    int cyclesUsed = 0;
    auto t0 = Clock::now();
    vector<double> hist = solver.solve(u, f, tol, maxCycles, cyclesUsed);
    auto t1 = Clock::now();

    cout << "\nMultigrid Convergence  (n=" << n0 << ", levels=" << solver.depth()
         << ", " << smootherName(p.smoother) << ", " << cycleName(p.cycleType) << ")\n";
    cout << "---------------------------------------------------------------\n";
    for (size_t c = 0; c < hist.size(); c++)
        cout << "  cycle " << setw(3) << c << "  residual = " << hist[c] << "\n";

    double err = l2ErrorInterior(u, uExact, n0);
    double rho = convergenceFactor(hist);

    cout << "\nCycles to tolerance : " << cyclesUsed << "\n";
    cout << "Final L2 error vs analytical solution : " << err << "\n";
    cout << "Asymptotic convergence factor (per cycle) : " << rho << "\n";
    cout << "Wall time : " << chrono::duration<double, milli>(t1 - t0).count() << " ms\n";
}

//=====================================================
// [3] Automatic benchmark suite
// Runs a grid of configurations with NO user input and
// writes a CSV to results/mg_benchmark.csv, so the same
// runs are reproducible for the report.
//=====================================================
struct BenchRow
{
    int k, n, levels;
    string smoother, cycle;
    double omega;
    int preSweeps, postSweeps;
    int cycles;
    double finalResidual;
    double finalError;
    double rho;
    double timeMs;
};

static BenchRow runOneConfig(
    int k, int numLevels, SmootherType sm, double omega,
    int preSweeps, int postSweeps, CycleType ct,
    double tol, int maxCycles)
{
    int n0 = nFromK(k);

    MGParams p;
    p.smoother = sm;
    p.omega = omega;
    p.preSweeps = preSweeps;
    p.postSweeps = postSweeps;
    p.cycleType = ct;
    p.numLevels = numLevels;

    MultigridSolver solver(n0, p);

    vector<double> f, uExact;
    buildProblem(n0, solver.finestH(), f, uExact);
    vector<double> u(n0 * n0, 0.0);

    int cyclesUsed = 0;
    auto t0 = Clock::now();
    vector<double> hist = solver.solve(u, f, tol, maxCycles, cyclesUsed);
    auto t1 = Clock::now();

    BenchRow row;
    row.k = k;
    row.n = n0;
    row.levels = solver.depth();
    row.smoother = smootherName(sm);
    row.cycle = cycleName(ct);
    row.omega = omega;
    row.preSweeps = preSweeps;
    row.postSweeps = postSweeps;
    row.cycles = cyclesUsed;
    row.finalResidual = hist.back();
    row.finalError = l2ErrorInterior(u, uExact, n0);
    row.rho = convergenceFactor(hist);
    row.timeMs = chrono::duration<double, milli>(t1 - t0).count();
    return row;
}

static void writeCsv(const string& path, const vector<BenchRow>& rows)
{
    ofstream out(path);
    out << "k,n,levels,smoother,cycle,omega,preSweeps,postSweeps,"
           "cycles,finalResidual,finalError,convFactor,timeMs\n";
    for (const auto& r : rows)
    {
        out << r.k << "," << r.n << "," << r.levels << "," << r.smoother << ","
            << r.cycle << "," << r.omega << "," << r.preSweeps << ","
            << r.postSweeps << "," << r.cycles << "," << r.finalResidual << ","
            << r.finalError << "," << r.rho << "," << r.timeMs << "\n";
    }
}

static void printTable(const vector<BenchRow>& rows, const string& title)
{
    cout << "\n" << title << "\n";
    cout << string(title.size(), '=') << "\n";
    cout << left
         << setw(4) << "k" << setw(6) << "n" << setw(4) << "Lv"
         << setw(18) << "Smoother" << setw(9) << "Cycle"
         << setw(7) << "omega" << setw(6) << "pre" << setw(6) << "post"
         << setw(9) << "cycles" << setw(13) << "residual"
         << setw(13) << "L2 error" << setw(10) << "rho"
         << setw(10) << "time(ms)" << "\n";

    for (const auto& r : rows)
    {
        cout << left
             << setw(4) << r.k << setw(6) << r.n << setw(4) << r.levels
             << setw(18) << r.smoother << setw(9) << r.cycle
             << setw(7) << r.omega << setw(6) << r.preSweeps << setw(6) << r.postSweeps
             << setw(9) << r.cycles << setw(13) << r.finalResidual
             << setw(13) << r.finalError << setw(10) << r.rho
             << setw(10) << r.timeMs << "\n";
    }
}

static void runBenchmarkSuite()
{
    const double tol = 1e-8;
    const int maxCycles = 50;

    vector<BenchRow> smootherCycleTable;
    vector<BenchRow> meshIndependenceTable;
    vector<BenchRow> levelsDepthTable;

    //-------------------------------------------------
    // A) Smoother x Cycle-type comparison, fixed grid k=6 (n=65)
    //-------------------------------------------------
    int kFixed = 6;
    struct SmCfg { SmootherType s; double omega; string label; };
    vector<SmCfg> smCfgs = {
        {SmootherType::JACOBI,          1.0,  "Jacobi"},
        {SmootherType::WEIGHTED_JACOBI, 0.8,  "W-Jacobi(0.8)"},
        {SmootherType::GAUSS_SEIDEL,    1.0,  "Gauss-Seidel"},
        {SmootherType::SOR,             1.5,  "SOR(1.5)"},
    };
    vector<CycleType> cycles = {CycleType::V, CycleType::W};

    for (auto& sc : smCfgs)
        for (auto ct : cycles)
            smootherCycleTable.push_back(
                runOneConfig(kFixed, -1, sc.s, sc.omega, 2, 2, ct, tol, maxCycles));

    printTable(smootherCycleTable, "A) Smoother x Cycle-type comparison (n=65, full V/W depth)");

    //-------------------------------------------------
    // B) Mesh-independence study: fixed config (GS, V-cycle,
    //    2 pre/2 post, full depth), grid size k = 4..8.
    //    Multigrid theory predicts roughly CONSTANT cycle
    //    count regardless of problem size.
    //-------------------------------------------------
    for (int k = 4; k <= 8; k++)
        meshIndependenceTable.push_back(
            runOneConfig(k, -1, SmootherType::GAUSS_SEIDEL, 1.0, 2, 2, CycleType::V, tol, maxCycles));

    printTable(meshIndependenceTable, "B) Mesh-independence study (Gauss-Seidel, V-cycle, full depth)");

    //-------------------------------------------------
    // C) Effect of hierarchy depth (number of levels),
    //    fixed grid k=7 (n=129), GS smoother, V-cycle.
    //-------------------------------------------------
    for (int nl = 1; nl <= 7; nl++)
        levelsDepthTable.push_back(
            runOneConfig(7, nl, SmootherType::GAUSS_SEIDEL, 1.0, 2, 2, CycleType::V, tol, maxCycles));

    printTable(levelsDepthTable, "C) Effect of hierarchy depth (n=129, Gauss-Seidel, V-cycle)");

    //-------------------------------------------------
    // Write combined CSV for the report
    //-------------------------------------------------
    vector<BenchRow> all;
    all.insert(all.end(), smootherCycleTable.begin(), smootherCycleTable.end());
    all.insert(all.end(), meshIndependenceTable.begin(), meshIndependenceTable.end());
    all.insert(all.end(), levelsDepthTable.begin(), levelsDepthTable.end());

    writeCsv("results/mg_benchmark.csv", all);
    cout << "\nFull results written to results/mg_benchmark.csv\n";
}

//=====================================================
// Main
//=====================================================
int main()
{
    cout << "=====================================\n";
    cout << "   MULTIGRID POISSON SOLVER\n";
    cout << "=====================================\n";
    cout << "[1] Legacy single-grid demo (Assignment 1-4 solvers)\n";
    cout << "[2] Interactive multigrid solve (choose everything yourself)\n";
    cout << "[3] Automatic benchmark suite (no input, writes CSV for report)\n";
    cout << "Choice : ";

    int choice;
    cin >> choice;

    if (choice == 1)
        runLegacyDemo();
    else if (choice == 2)
        runInteractiveMultigrid();
    else
        runBenchmarkSuite();

    return 0;
}
