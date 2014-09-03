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

#ifndef POLYQUAD_TET_HPP_
#define POLYQUAD_TET_HPP_

#include "base.hpp"
#include "jacobi_poly.hpp"
#include "util.hpp"

#include <Eigen/Dense>

#include <cassert>

namespace polyquad {

template<typename T>
class TetDomain : public BaseDomain<TetDomain<T>, T, 3, 5>
{
public:
    typedef BaseDomain<TetDomain<T>, T, 3, 5> Base;
    typedef typename Base::MatrixXT MatrixXT;
    typedef typename Base::VectorXT VectorXT;
    typedef typename Base::MatrixPtsT MatrixPtsT;
    typedef typename Base::VectorOrb VectorOrb;

    typedef Eigen::Matrix<T, 3, 1> Vector3T;

public:
    TetDomain() : Base(2/sqrt(T(3)))
    {}

    bool validate_orbit(const VectorOrb& orb) const
    { return orb(0) <= 1; }

private:
    friend class BaseDomain<TetDomain<T>, T, 3, 5>;

    int npts_for_orbit(int i) const;

    int narg_for_orbit(int i) const;

    int nbfn_for_qdeg(int qdeg) const;

    void expand_orbit(int i, int aoff, int poff, const VectorXT& args,
                      MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pqr, D2 out) const;

    void clamp_arg(int i, int aoff, VectorXT& args) const;

private:
    Vector3T bary_to_cart(const T& p1, const T& p2, const T& p3,
                          const T& p4) const
    { return {-p4 - p3 + p2 - p1, -p4 + p3 - p2 - p1, p4 - p3 - p2 - p1}; }
};

template<typename T>
inline int
TetDomain<T>::npts_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 1;
        case 1:
            return 4;
        case 2:
            return 6;
        case 3:
            return 12;
        case 4:
            return 24;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline int
TetDomain<T>::narg_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 0;
        case 1:
        case 2:
            return 1;
        case 3:
            return 2;
        case 4:
            return 3;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline int
TetDomain<T>::nbfn_for_qdeg(int qdeg) const
{
    int n = 0;

    for (int i = 0; i <= qdeg; ++i)
        for (int j = i; j <= qdeg - i; ++j)
            for (int k = j; k <= qdeg - i - j; ++k, ++n);

    return n;
}

template<typename T>
EIGEN_ALWAYS_INLINE void
TetDomain<T>::expand_orbit(int i, int aoff, int poff,
                           const VectorXT& args, MatrixPtsT& pts) const
{
    switch (i)
    {
        case 0:
        {
            const T& a = T(1) / 4;
            pts.row(poff) = bary_to_cart(a, a, a, a);
            break;
        }
        case 1:
        {
            const T& a = args(aoff);
            pts.row(poff + 0) = bary_to_cart(a, a, a, 1 - 3*a);
            pts.row(poff + 1) = bary_to_cart(a, a, 1 - 3*a, a);
            pts.row(poff + 2) = bary_to_cart(a, 1 - 3*a, a, a);
            pts.row(poff + 3) = bary_to_cart(1 - 3*a, a, a, a);
            break;
        }
        case 2:
        {
            const T& a = args(aoff);
            pts.row(poff + 0) = bary_to_cart(a, a, 0.5 - a, 0.5 - a);
            pts.row(poff + 1) = bary_to_cart(a, 0.5 - a, a, 0.5 - a);
            pts.row(poff + 2) = bary_to_cart(0.5 - a, a, a, 0.5 - a);
            pts.row(poff + 3) = bary_to_cart(0.5 - a, a, 0.5 - a, a);
            pts.row(poff + 4) = bary_to_cart(0.5 - a, 0.5 - a, a, a);
            pts.row(poff + 5) = bary_to_cart(a, 0.5 - a, 0.5 - a, a);
            break;
        }
        case 3:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = bary_to_cart(a, 1 - 2*a - b, a, b);
            pts.row(poff + 1) = bary_to_cart(b, 1 - 2*a - b, a, a);
            pts.row(poff + 2) = bary_to_cart(b, a, a, 1 - 2*a - b);
            pts.row(poff + 3) = bary_to_cart(a, b, 1 - 2*a - b, a);
            pts.row(poff + 4) = bary_to_cart(a, a, b, 1 - 2*a - b);
            pts.row(poff + 5) = bary_to_cart(b, a, 1 - 2*a - b, a);
            pts.row(poff + 6) = bary_to_cart(a, b, a, 1 - 2*a - b);
            pts.row(poff + 7) = bary_to_cart(1 - 2*a - b, a, b, a);
            pts.row(poff + 8) = bary_to_cart(1 - 2*a - b, a, a, b);
            pts.row(poff + 9) = bary_to_cart(a, a, 1 - 2*a - b, b);
            pts.row(poff + 10) = bary_to_cart(1 - 2*a - b, b, a, a);
            pts.row(poff + 11) = bary_to_cart(a, 1 - 2*a - b, b, a);
            break;
        }
        case 4:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            const T& c = args(aoff + 2);
            pts.row(poff + 0) = bary_to_cart(c, 1 - a - b - c, b, a);
            pts.row(poff + 1) = bary_to_cart(c, b, a, 1 - a - b - c);
            pts.row(poff + 2) = bary_to_cart(c, b, 1 - a - b - c, a);
            pts.row(poff + 3) = bary_to_cart(b, a, 1 - a - b - c, c);
            pts.row(poff + 4) = bary_to_cart(c, a, 1 - a - b - c, b);
            pts.row(poff + 5) = bary_to_cart(b, c, 1 - a - b - c, a);
            pts.row(poff + 6) = bary_to_cart(1 - a - b - c, a, b, c);
            pts.row(poff + 7) = bary_to_cart(b, a, c, 1 - a - b - c);
            pts.row(poff + 8) = bary_to_cart(1 - a - b - c, c, b, a);
            pts.row(poff + 9) = bary_to_cart(a, 1 - a - b - c, b, c);
            pts.row(poff + 10) = bary_to_cart(a, c, 1 - a - b - c, b);
            pts.row(poff + 11) = bary_to_cart(c, a, b, 1 - a - b - c);
            pts.row(poff + 12) = bary_to_cart(a, c, b, 1 - a - b - c);
            pts.row(poff + 13) = bary_to_cart(b, 1 - a - b - c, c, a);
            pts.row(poff + 14) = bary_to_cart(1 - a - b - c, a, c, b);
            pts.row(poff + 15) = bary_to_cart(1 - a - b - c, b, c, a);
            pts.row(poff + 16) = bary_to_cart(a, 1 - a - b - c, c, b);
            pts.row(poff + 17) = bary_to_cart(a, b, c, 1 - a - b - c);
            pts.row(poff + 18) = bary_to_cart(a, b, 1 - a - b - c, c);
            pts.row(poff + 19) = bary_to_cart(b, c, a, 1 - a - b - c);
            pts.row(poff + 20) = bary_to_cart(b, 1 - a - b - c, a, c);
            pts.row(poff + 21) = bary_to_cart(c, 1 - a - b - c, a, b);
            pts.row(poff + 22) = bary_to_cart(1 - a - b - c, b, a, c);
            pts.row(poff + 23) = bary_to_cart(1 - a - b - c, c, a, b);
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline void
TetDomain<T>::seed_orbit(int i, int aoff, VectorXT& args)
{
    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = this->rand(0.0, 1.0 / 3.0);
            break;
        case 2:
            args(aoff) = this->rand(0.0, 0.5);
            break;
        case 3:
            args(aoff + 0) = this->rand(0.0, 1.0 / 3.0);
            args(aoff + 1) = this->rand(0.0, 1.0 / 3.0);
            break;
        case 4:
            args(aoff + 0) = this->rand(0.0, 0.25);
            args(aoff + 1) = this->rand(0.0, 0.25);
            args(aoff + 2) = this->rand(0.0, 0.25);
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
template<typename D1, typename D2>
inline void
TetDomain<T>::eval_orthob_block(const D1 pqr, D2 out) const
{
    typedef Eigen::Array<T, D1::RowsAtCompileTime, 1> ArrayT;

    const auto& p = pqr.col(0);
    const auto& q = pqr.col(1);
    const auto& r = pqr.col(2);

    const auto& a = (q != -r).select(-2*(1 + p)/(q + r) - 1, 0);
    const auto& b = (r != 1).select(2*(1 + q)/(1 - r) - 1, 0);
    const auto& c = r;

    const T one = 1;

    ArrayT pow1mbi = ArrayT::Constant(p.size(), 1);
    ArrayT pow1mci = ArrayT::Constant(p.size(), 1);

    for (int i = 0, off = 0; i <= this->qdeg(); ++i)
    {
        T ci = exp2(T(-2*i - 1.5))*sqrt(T(4*i + 2));

        ArrayT pow1mcj = pow1mci;

        for (int j = i; j <= this->qdeg() - i; ++j)
        {
            T cij = ci*sqrt(T(i + j + 1))*ldexp(one, -j);

            for (int k = j; k <= this->qdeg() - i - j; ++k, ++off)
            {
                T cijk = cij*sqrt(T(2*(k + j + i) + 3));

                out.row(off) = cijk*pow1mbi*pow1mci*pow1mcj
                             * jacobi_poly(i, 0, 0, a)
                             * jacobi_poly(j, 2*i + 1, 0, b)
                             * jacobi_poly(k, 2*(i + j + 1), 0, c);
            }

            pow1mcj *= 1 - c;
        }

        pow1mbi *= 1 - b;
        pow1mci *= 1 - c;
    }
}

template<typename T>
inline void
TetDomain<T>::clamp_arg(int i, int aoff, VectorXT& args) const
{
    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = clamp(0, args(aoff), T(1) / 3);
            break;
        case 2:
            args(aoff) = clamp(0, args(aoff), 0.5);
            break;
        case 3:
            args(aoff + 0) = clamp(0, args(aoff + 0), 0.5);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1 - 2*args(aoff + 0));
            break;
        case 4:
            args(aoff + 0) = clamp(0, args(aoff + 0), 1);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1 - args(aoff + 0));
            args(aoff + 2) = clamp(0, args(aoff + 2),
                                   1 - args(aoff + 0) - args(aoff + 1));
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

}

#endif /* POLYQUAD_TET_HPP_ */
