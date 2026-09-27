#ifndef SMOOTHERS_H
#define SMOOTHERS_H

#include <vector>

void Jacobi(
    const std::vector<std::vector<double>>& A,
    std::vector<double>& x,
    const std::vector<double>& b,
    int maxIter);

void GaussSeidel(
    const std::vector<std::vector<double>>& A,
    std::vector<double>& x,
    const std::vector<double>& b,
    int maxIter);

void SOR(
    const std::vector<std::vector<double>>& A,
    std::vector<double>& x,
    const std::vector<double>& b,
    int maxIter,
    double omega);

#endif