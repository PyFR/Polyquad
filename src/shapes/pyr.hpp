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

#ifndef POLYQUAD_SHAPES_PYR_HPP
#define POLYQUAD_SHAPES_PYR_HPP

#include "shapes/base.hpp"
#include "utils/ortho_poly.hpp"

#include <Eigen/Dense>

#include <cassert>

namespace polyquad {

template<typename T>
class PyrDomain : public BaseDomain<PyrDomain<T>, T, 3, 4>
{
public:
    typedef BaseDomain<PyrDomain<T>, T, 3, 4> Base;
    typedef typename Base::MatrixXT MatrixXT;
    typedef typename Base::VectorXT VectorXT;
    typedef typename Base::MatrixPtsT MatrixPtsT;
    typedef typename Base::VectorOrb VectorOrb;

    typedef Eigen::Matrix<T, 3, 1> Vector3T;

public:
    PyrDomain() : Base(2*sqrt(T(6))/3)
    {}

    bool validate_orbit(const VectorOrb&) const
    { return true; }

private:
    friend class BaseDomain<PyrDomain<T>, T, 3, 4>;

    constexpr int npts_for_orbit(int i) const;

    constexpr int narg_for_orbit(int i) const;

    constexpr int nbfn_for_qdeg(int qdeg) const;

    void expand_orbit(int i, int aoff, int poff,
                      const VectorXT& args,
                      MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pqr, D2 out) const;

    void clamp_arg(int i, int aoff, VectorXT& args) const;
};

template<typename T>
inline constexpr int
PyrDomain<T>::npts_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 1;
        case 1:
        case 2:
            return 4;
        case 3:
            return 8;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline constexpr int
PyrDomain<T>::narg_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 1;
        case 1:
        case 2:
            return 2;
        case 3:
            return 3;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline constexpr int
PyrDomain<T>::nbfn_for_qdeg(int qdeg) const
{
    int n = 0;

    for (int i = 0; i <= qdeg; i += 2)
        for (int j = i; j <= qdeg - i; j += 2)
            for (int k = 0; k <= qdeg - i - j; ++k, ++n);

    return n;
}

template<typename T>
EIGEN_ALWAYS_INLINE void
PyrDomain<T>::expand_orbit(int i, int aoff, int poff,
                           const VectorXT& args, MatrixPtsT& pts) const
{
    switch (i)
    {
        case 0:
        {
            pts.row(poff) = Vector3T(0, 0, args(aoff));
            break;
        }
        case 1:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = Vector3T(a, 0, b);
            pts.row(poff + 1) = Vector3T(0, a, b);
            pts.row(poff + 2) = Vector3T(-a, 0, b);
            pts.row(poff + 3) = Vector3T(0, -a, b);
            break;
        }
        case 2:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = Vector3T(a, a, b);
            pts.row(poff + 1) = Vector3T(a, -a, b);
            pts.row(poff + 2) = Vector3T(-a, a, b);
            pts.row(poff + 3) = Vector3T(-a, -a, b);
            break;
        }
        case 3:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            const T& c = args(aoff + 2);
            pts.row(poff + 0) = Vector3T(a, b, c);
            pts.row(poff + 1) = Vector3T(b, a, c);
            pts.row(poff + 2) = Vector3T(a, -b, c);
            pts.row(poff + 3) = Vector3T(-b, a, c);
            pts.row(poff + 4) = Vector3T(-a, b, c);
            pts.row(poff + 5) = Vector3T(b, -a, c);
            pts.row(poff + 6) = Vector3T(-a, -b, c);
            pts.row(poff + 7) = Vector3T(-b, -a, c);
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline void
PyrDomain<T>::seed_orbit(int i, int aoff, VectorXT& args)
{
    switch (i)
    {
        case 0:
            args(aoff + 0) = this->rand(-1.0, 1.0);
            break;
        case 1:
        case 2:
        {
            double b = this->rand(-1.0, 1.0);
            args(aoff + 0) = this->rand(0.0, (1.0 - b)/2);
            args(aoff + 1) = b;
            break;
        }
        case 3:
        {
            double c = this->rand(-1.0, 1.0);
            args(aoff + 0) = this->rand(0.0, (1.0 - c)/2);
            args(aoff + 1) = this->rand(0.0, (1.0 - c)/2);
            args(aoff + 2) = c;
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
template<typename D1, typename D2>
inline void
PyrDomain<T>::eval_orthob_block(const D1 pqr, D2 out) const
{
    typedef Eigen::Array<T, D1::RowsAtCompileTime, 1> ArrayT;

    const auto& p = pqr.col(0);
    const auto& q = pqr.col(1);
    const auto& r = pqr.col(2);

    const ArrayT a = (r != 1).select(2*p/(1 - r), 0);
    const ArrayT b = (r != 1).select(2*q/(1 - r), 0);
    const auto& c = r;

    const T half = 0.5;
    const T pow2m12 = sqrt(half);
    T pow2mi = 1;

    ArrayT pow1mci = ArrayT::Constant(p.size(), 1);

    EvenLegendreP<ArrayT> jpa(a);

    for (int i = 0, off = 0; i <= this->qdeg(); i += 2)
    {
        T pow2mj = pow2mi;
        ArrayT pow1mcj = pow1mci;
        EvenLegendreP<ArrayT> jpb(b);

        for (int j = i; j <= this->qdeg() - i; j += 2)
        {
            T cij = pow2m12*pow2mi*pow2mj;
            JacobiP<ArrayT> jpc(2*(i + j + 1), 0, c);

            for (int k = 0; k <= this->qdeg() - i - j; ++k, ++off)
            {
                T cijk = cij*sqrt((2*(k + j + i) + 3)*(i + half)*(j + half));

                out.row(off) = cijk*pow1mci*pow1mcj*jpa(i)*jpb(j)*jpc(k);
            }

            pow2mj /= 4;
            pow1mcj *= (1 - c)*(1 - c);
        }

        pow2mi /= 4;
        pow1mci *= (1 - c)*(1 - c);
    }
}

template<typename T>
inline void
PyrDomain<T>::clamp_arg(int i, int aoff, VectorXT& args) const
{
    switch (i)
    {
        case 0:
            args(aoff + 0) = clamp(-1, args(aoff), 1);
            break;
        case 1:
        case 2:
            args(aoff + 1) = clamp(-1, args(aoff + 1), 1);
            args(aoff + 0) = clamp(0, args(aoff + 0), (1 - args(aoff + 1))/2);
            break;
        case 3:
        {
            args(aoff + 2) = clamp(-1, args(aoff + 2), 1);
            args(aoff + 1) = clamp(0, args(aoff + 1), (1 - args(aoff + 2))/2);
            args(aoff + 0) = clamp(0, args(aoff + 0), (1 - args(aoff + 2))/2);
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

}

#endif /* POLYQUAD_SHAPES_PYR_HPP */
