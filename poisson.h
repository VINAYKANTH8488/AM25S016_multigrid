#ifndef POISSON_H
#define POISSON_H

#include <vector>

int idx(int i,int j,int nx);

double exact(double x,double y);

double rhs(double x,double y);

std::vector<std::vector<double>>
buildPoissonMatrix(int nx,int ny);

#endif