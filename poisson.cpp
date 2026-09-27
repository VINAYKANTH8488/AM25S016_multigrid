#include "poisson.h"

#include <cmath>
#include <vector>

using namespace std;

int idx(int i,int j,int nx)
{
    return j*nx+i;
}

double exact(double x,double y)
{
    return sin(M_PI*x)*sin(M_PI*y);
}

double rhs(double x,double y)
{
    return 2.0*M_PI*M_PI*
           sin(M_PI*x)*
           sin(M_PI*y);
}

vector<vector<double>>
buildPoissonMatrix(int nx,int ny)
{
    int N = nx*ny;

    double hx = 1.0/(nx+1);
    double hy = 1.0/(ny+1);

    double ax = 1.0/(hx*hx);
    double ay = 1.0/(hy*hy);

    vector<vector<double>>
    A(N,vector<double>(N,0.0));

    for(int j=0;j<ny;j++)
    {
        for(int i=0;i<nx;i++)
        {
            int p=idx(i,j,nx);

            A[p][p]=2.0*(ax+ay);

            if(i>0)
                A[p][idx(i-1,j,nx)] = -ax;

            if(i<nx-1)
                A[p][idx(i+1,j,nx)] = -ax;

            if(j>0)
                A[p][idx(i,j-1,nx)] = -ay;

            if(j<ny-1)
                A[p][idx(i,j+1,nx)] = -ay;
        }
    }

    return A;
}