/*
    This file is part of polyquad.
    Copyright (C) 2014  Freddie Witherden <freddie@witherden.org>

    Polyquad is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    Polyquad is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with polyquad.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef POLYQUAD_UTILS_JACOBI_POLY_HPP
#define POLYQUAD_UTILS_JACOBI_POLY_HPP

#include <Eigen/Dense>

namespace polyquad {

template<typename ArrayT>
class JacobiP
{
public:
    typedef typename ArrayT::Scalar T;

public:
    JacobiP(int a, int b, const ArrayT& x) : a_(a), b_(b), q_(0), x_(x)
    {}

    ArrayT operator()(int n);

private:
    const int a_, b_;
    int q_;

    const ArrayT x_;
    ArrayT jm1_, jm2_;
};

template<typename ArrayT>
EIGEN_ALWAYS_INLINE ArrayT
JacobiP<ArrayT>::operator()(int n)
{
    assert(q_ - 1 <= n && "Polynomials must be evaluated in sequence");

    int apb = a_ + b_, amb = a_ - b_, bbmaa = b_*b_ - a_*a_;

    for (; q_ <= n; ++q_)
    {
        if (q_ == 0)
            jm1_ = ArrayT::Constant(x_.rows(), x_.cols(), 1);
        else if (q_ == 1)
        {
            jm2_ = jm1_;
            jm1_ = ((apb + 2)*x_ + amb) / 2;
        }
        else
        {
            int qapbpq = q_*(apb + q_), apbp2q = apb + 2*q_;
            int apbp2qm1 = apbp2q - 1, apbp2qm2 = apbp2q - 2;

            T aq = T(apbp2q*apbp2qm1) / (2*qapbpq);
            T bq = T(apbp2qm1*bbmaa) / (2*qapbpq*apbp2qm2);
            T cq = T(apbp2q)*((a_ + q_ - 1)*(b_ + q_ - 1)) / (qapbpq*apbp2qm2);

            std::swap(jm1_, jm2_);
            jm1_ = (aq*x_ - bq)*jm2_ - cq*jm1_;
        }
    }

    return jm1_;
}

}

#endif /* POLYQUAD_UTILS_JACOBI_POLY_HPP */
