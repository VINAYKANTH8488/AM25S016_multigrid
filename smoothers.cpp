#include "smoothers.h"

#include <vector>

using namespace std;

void Jacobi(
    const vector<vector<double>>& A,
    vector<double>& x,
    const vector<double>& b,
    int maxIter)
{
    int N=A.size();

    vector<double> xnew(N,0.0);

    for(int iter=0;iter<maxIter;iter++)
    {
        for(int i=0;i<N;i++)
        {
            double sigma=0.0;

            for(int j=0;j<N;j++)
            {
                if(i!=j)
                    sigma += A[i][j]*x[j];
            }

            xnew[i] =
                (b[i]-sigma)/A[i][i];
        }

        x=xnew;
    }
}

void GaussSeidel(
    const vector<vector<double>>& A,
    vector<double>& x,
    const vector<double>& b,
    int maxIter)
{
    int N=A.size();

    for(int iter=0;iter<maxIter;iter++)
    {
        for(int i=0;i<N;i++)
        {
            double sigma=0.0;

            for(int j=0;j<N;j++)
            {
                if(i!=j)
                    sigma += A[i][j]*x[j];
            }

            x[i] =
                (b[i]-sigma)/A[i][i];
        }
    }
}

void SOR(
    const vector<vector<double>>& A,
    vector<double>& x,
    const vector<double>& b,
    int maxIter,
    double omega)
{
    int N=A.size();

    for(int iter=0;iter<maxIter;iter++)
    {
        for(int i=0;i<N;i++)
        {
            double sigma=0.0;

            for(int j=0;j<N;j++)
            {
                if(i!=j)
                    sigma += A[i][j]*x[j];
            }

            double xgs =
                (b[i]-sigma)/A[i][i];

            x[i] =
                (1.0-omega)*x[i]
                + omega*xgs;
        }
    }
}