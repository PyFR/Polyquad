
#ifndef POLYQUAD_JACOBI_POLY_HPP
#define POLYQUAD_JACOBI_POLY_HPP

#include <Eigen/Dense>

namespace polyquad {


template<typename D>
Eigen::Array<typename D::Scalar, D::RowsAtCompileTime, D::ColsAtCompileTime>
jacobi_poly(int n, int a, int b, const Eigen::DenseBase<D>& x)
{
    typedef typename D::Scalar T;
    typedef Eigen::Array<T, D::RowsAtCompileTime, D::ColsAtCompileTime> ArrayT;

    ArrayT xv = x;
    int apb = a + b, amb = a - b, bbmaa = b*b - a*a;

    if (n == 0)
        return ArrayT::Constant(x.rows(), x.cols(), 1);
    else if (n == 1)
        return ((apb + 2)*xv + amb) / 2;
    else
    {
        ArrayT jm1 = ((apb + 2)*xv + amb) / 2;
        ArrayT jm2 = ArrayT::Constant(x.rows(), x.cols(), 1);

        for (int q = 2; q <= n; ++q)
        {
            int qapbpq = q*(apb + q), apbp2q = apb + 2*q;
            int apbp2qm1 = apbp2q - 1, apbp2qm2 = apbp2q - 2;

            T aq = T(apbp2q*apbp2qm1) / (2*qapbpq);
            T bq = T(apbp2qm1*bbmaa) / (2*qapbpq*apbp2qm2);
            T cq = T(apbp2q)*((a + q - 1)*(b + q - 1)) / (qapbpq*apbp2qm2);

            std::swap(jm1, jm2);
            jm1 = (aq*xv - bq)*jm2 - cq*jm1;
        }

        return jm1;
    }
}

}

#endif /* POLYQUAD_JACOBI_POLY_HPP */
