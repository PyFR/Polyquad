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

#ifndef POLYQUAD_SHAPES_PRI_HPP
#define POLYQUAD_SHAPES_PRI_HPP

#include "shapes/base.hpp"
#include "utils/jacobi_poly.hpp"

#include <Eigen/Dense>

#include <cassert>

namespace polyquad {

template<typename T>
class PriDomain : public BaseDomain<PriDomain<T>, T, 3, 6>
{
public:
    typedef BaseDomain<PriDomain<T>, T, 3, 6> Base;
    typedef typename Base::MatrixXT MatrixXT;
    typedef typename Base::VectorXT VectorXT;
    typedef typename Base::MatrixPtsT MatrixPtsT;
    typedef typename Base::VectorOrb VectorOrb;

    typedef Eigen::Matrix<T, 3, 1> Vector3T;

public:
    PriDomain() : Base(2)
    {}

    bool validate_orbit(const VectorOrb& orb) const
    { return orb(0) <= 1; }

private:
    friend class BaseDomain<PriDomain<T>, T, 3, 6>;

    int npts_for_orbit(int i) const;

    int narg_for_orbit(int i) const;

    int nbfn_for_qdeg(int qdeg) const;

    void expand_orbit(int i, int aoff, int poff,
                      const VectorXT& args, MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pqr, D2 out) const;

    void clamp_arg(int i, int aoff, VectorXT& args) const;

private:
    Vector3T bary_to_cart(const T& p1, const T& p2, const T& p3,
                          const T& z) const
    { return {-p1 + p2 - p3, -p1 - p2 + p3, z}; }
};

template<typename T>
inline int
PriDomain<T>::npts_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 1;
        case 1:
            return 2;
        case 2:
            return 3;
        case 3:
        case 4:
            return 6;
        case 5:
            return 12;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline int
PriDomain<T>::narg_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 0;
        case 1:
        case 2:
            return 1;
        case 3:
        case 4:
            return 2;
        case 5:
            return 3;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline int
PriDomain<T>::nbfn_for_qdeg(int qdeg) const
{
    int n = 0;

    for (int i = 0; i <= qdeg; ++i)
        for (int j = i; j <= qdeg - i; ++j)
            for (int k = 0; k <= qdeg - i - j; k += 2, ++n);

    return n;
}

template<typename T>
EIGEN_ALWAYS_INLINE void
PriDomain<T>::expand_orbit(int i, int aoff, int poff,
                           const VectorXT& args, MatrixPtsT& pts) const
{
    switch (i)
    {
        case 0:
        {
            const T& a = static_cast<T>(1) / 3;
            pts.row(poff) = bary_to_cart(a, a, a, 0);
            break;
        }
        case 1:
        {
            const T& a = T(1) / 3;
            const T& b = args(aoff);
            pts.row(poff + 0) = bary_to_cart(a, a, a, -b);
            pts.row(poff + 1) = bary_to_cart(a, a, a, b);
            break;
        }
        case 2:
        {
            const T& a = args(aoff);
            pts.row(poff + 0) = bary_to_cart(a, a, 1 - 2*a, 0);
            pts.row(poff + 1) = bary_to_cart(a, 1 - 2*a, a, 0);
            pts.row(poff + 2) = bary_to_cart(1 - 2*a, a, a, 0);
            break;
        }
        case 3:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = bary_to_cart(a, a, 1 - 2*a, -b);
            pts.row(poff + 1) = bary_to_cart(a, 1 - 2*a, a, -b);
            pts.row(poff + 2) = bary_to_cart(1 - 2*a, a, a, -b);
            pts.row(poff + 3) = bary_to_cart(a, a, 1 - 2*a, b);
            pts.row(poff + 4) = bary_to_cart(a, 1 - 2*a, a, b);
            pts.row(poff + 5) = bary_to_cart(1 - 2*a, a, a, b);
            break;
        }
        case 4:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = bary_to_cart(a, b, 1 - a - b, 0);
            pts.row(poff + 1) = bary_to_cart(a, 1 - a - b, b, 0);
            pts.row(poff + 2) = bary_to_cart(b, a, 1 - a - b, 0);
            pts.row(poff + 3) = bary_to_cart(b, 1 - a - b, a, 0);
            pts.row(poff + 4) = bary_to_cart(1 - a - b, a, b, 0);
            pts.row(poff + 5) = bary_to_cart(1 - a - b, b, a, 0);
            break;
        }
        case 5:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            const T& c = args(aoff + 2);
            pts.row(poff + 0)  = bary_to_cart(a, b, 1 - a - b, -c);
            pts.row(poff + 1)  = bary_to_cart(a, 1 - a - b, b, -c);
            pts.row(poff + 2)  = bary_to_cart(b, a, 1 - a - b, -c);
            pts.row(poff + 3)  = bary_to_cart(b, 1 - a - b, a, -c);
            pts.row(poff + 4)  = bary_to_cart(1 - a - b, a, b, -c);
            pts.row(poff + 5)  = bary_to_cart(1 - a - b, b, a, -c);
            pts.row(poff + 6)  = bary_to_cart(a, b, 1 - a - b, c);
            pts.row(poff + 7)  = bary_to_cart(a, 1 - a - b, b, c);
            pts.row(poff + 8)  = bary_to_cart(b, a, 1 - a - b, c);
            pts.row(poff + 9)  = bary_to_cart(b, 1 - a - b, a, c);
            pts.row(poff + 10) = bary_to_cart(1 - a - b, a, b, c);
            pts.row(poff + 11) = bary_to_cart(1 - a - b, b, a, c);
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline void
PriDomain<T>::seed_orbit(int i, int aoff, VectorXT& args)
{
    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = this->rand(0.0, 1.0);
            break;
        case 2:
            args(aoff) = this->rand(0.0, 0.5);
            break;
        case 3:
            args(aoff + 0) = this->rand(0.0, 0.5);
            args(aoff + 1) = this->rand(0.0, 1.0);
            break;
        case 4:
            args(aoff + 0) = this->rand(0.0, 1.0 / 3.0);
            args(aoff + 1) = this->rand(0.0, 1.0 / 3.0);
            break;
        case 5:
            args(aoff + 0) = this->rand(0.0, 1.0 / 3.0);
            args(aoff + 1) = this->rand(0.0, 1.0 / 3.0);
            args(aoff + 2) = this->rand(0.0, 1.0);
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
template<typename D1, typename D2>
inline void
PriDomain<T>::eval_orthob_block(const D1 pqr, D2 out) const
{
    typedef Eigen::Array<T, D1::RowsAtCompileTime, 1> ArrayT;

    const T half = 0.5;

    const auto& p = pqr.col(0);
    const auto& q = pqr.col(1);
    const auto& r = pqr.col(2);

    const auto& a = (q != 1).select(2*(1 + p)/(1 - q) - 1, 0);
    const auto& b = q;
    const auto& c = r;

    ArrayT pow1mqi = ArrayT::Constant(p.size(), 1);
    T pow2ip1 = half;

    JacobiP<ArrayT> jpa(0, 0, a);

    for (int i = 0, off = 0; i <= this->qdeg(); ++i)
    {
        JacobiP<ArrayT> jpb(2*i + 1, 0, b);

        for (int j = i; j <= this->qdeg() - i; ++j)
        {
            T cij = sqrt(T((2*i + 1)*(2*i + 2*j + 2)))*pow2ip1;
            JacobiP<ArrayT> jpc(0, 0, c);

            for (int k = 0; k <= this->qdeg() - i - j; k += 2, ++off)
            {
                T cijk = cij*sqrt(k + half);

                out.row(off) = cijk*pow1mqi*jpa(i)*jpb(j)*jpc(k);
            }
        }

        pow1mqi *= 1 - b;
        pow2ip1 /= 2;
    }
}

template<typename T>
inline void
PriDomain<T>::clamp_arg(int i, int aoff, VectorXT& args) const
{
    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = clamp(0, args(aoff), 1);
            break;
        case 2:
            args(aoff) = clamp(0, args(aoff), 0.5);
            break;
        case 3:
            args(aoff + 0) = clamp(0, args(aoff + 0), 0.5);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1);
            break;
        case 4:
            args(aoff + 0) = clamp(0, args(aoff + 0), 1);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1 - args(aoff + 0));
            break;
        case 5:
            args(aoff + 0) = clamp(0, args(aoff + 0), 1);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1 - args(aoff + 0));
            args(aoff + 2) = clamp(0, args(aoff + 2), 1);
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

}

#endif /* POLYQUAD_SHAPES_PRI_HPP */
