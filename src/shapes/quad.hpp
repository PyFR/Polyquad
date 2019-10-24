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

#ifndef POLYQUAD_SHAPES_QUAD_HPP
#define POLYQUAD_SHAPES_QUAD_HPP

#include "shapes/base.hpp"
#include "utils/jacobi_poly.hpp"

#include <Eigen/Dense>

#include <cassert>

namespace polyquad {

template<typename T>
class QuadDomain : public BaseDomain<QuadDomain<T>, T, 2, 4>
{
public:
    typedef BaseDomain<QuadDomain<T>, T, 2, 4> Base;
    typedef typename Base::MatrixXT MatrixXT;
    typedef typename Base::VectorXT VectorXT;
    typedef typename Base::MatrixPtsT MatrixPtsT;
    typedef typename Base::VectorOrb VectorOrb;

    typedef Eigen::Matrix<T, 2, 1> Vector2T;

public:
    QuadDomain() : Base(2)
    {}

    bool validate_orbit(const VectorOrb& orb) const
    { return orb(0) <= 1; }

private:
    friend class BaseDomain<QuadDomain<T>, T, 2, 4>;

    constexpr int npts_for_orbit(int i) const;

    constexpr int narg_for_orbit(int i) const;

    constexpr int nbfn_for_qdeg(int qdeg) const;

    void expand_orbit(int i, int aoff, int poff,
                      const VectorXT& args,
                      MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pq, D2 out) const;

    void clamp_arg(int i, int aoff, VectorXT& args) const;
};

template<typename T>
inline constexpr int
QuadDomain<T>::npts_for_orbit(int i) const
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
QuadDomain<T>::narg_for_orbit(int i) const
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
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline constexpr int
QuadDomain<T>::nbfn_for_qdeg(int qdeg) const
{
    int n = 0;

    for (int i = 0; i <= qdeg; i += 2)
        for (int j = i; j <= qdeg - i; j += 2, ++n);

    return n;
}

template<typename T>
EIGEN_ALWAYS_INLINE void
QuadDomain<T>::expand_orbit(int i, int aoff, int poff,
                            const VectorXT& args, MatrixPtsT& pts) const
{
    switch (i)
    {
        case 0:
        {
            pts.row(poff) = Vector2T(0, 0);
            break;
        }
        case 1:
        {
            const T& a = args(aoff);
            pts.row(poff + 0) = Vector2T(a, 0);
            pts.row(poff + 1) = Vector2T(0, a);
            pts.row(poff + 2) = Vector2T(-a, 0);
            pts.row(poff + 3) = Vector2T(0, -a);
            break;
        }
        case 2:
        {
            const T& a = args(aoff);
            pts.row(poff + 0) = Vector2T(a, a);
            pts.row(poff + 1) = Vector2T(a, -a);
            pts.row(poff + 2) = Vector2T(-a, a);
            pts.row(poff + 3) = Vector2T(-a, -a);
            break;
        }
        case 3:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = Vector2T(a, b);
            pts.row(poff + 1) = Vector2T(b, a);
            pts.row(poff + 2) = Vector2T(a, -b);
            pts.row(poff + 3) = Vector2T(-b, a);
            pts.row(poff + 4) = Vector2T(-a, b);
            pts.row(poff + 5) = Vector2T(b, -a);
            pts.row(poff + 6) = Vector2T(-a, -b);
            pts.row(poff + 7) = Vector2T(-b, -a);
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline void
QuadDomain<T>::seed_orbit(int i, int aoff, VectorXT& args)
{
    switch (i)
    {
        case 0:
            break;
        case 1:
        case 2:
            args(aoff) = this->rand(0.0, 1.0);
            break;
        case 3:
            args(aoff + 0) = this->rand(0.0, 1.0);
            args(aoff + 1) = this->rand(0.0, 1.0);
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
template<typename D1, typename D2>
inline void
QuadDomain<T>::eval_orthob_block(const D1 pq, D2 out) const
{
    typedef Eigen::Array<T, D1::RowsAtCompileTime, 1> ArrayT;

    const T half = 0.5;

    const auto& p = pq.col(0);
    const auto& q = pq.col(1);

    JacobiP<ArrayT> jpp(0, 0, p);

    for (int i = 0, off = 0; i <= this->qdeg(); i += 2)
    {
        JacobiP<ArrayT> jpq(0, 0, q);

        for (int j = i; j <= this->qdeg() - i; j += 2, ++off)
        {
            T cij = sqrt((i + half)*(j + half));

            out.row(off) = cij*jpp(i)*jpq(j);
        }
    }
}

template<typename T>
inline void
QuadDomain<T>::clamp_arg(int i, int aoff, VectorXT& args) const
{
    switch (i)
    {
        case 0:
            break;
        case 1:
        case 2:
            args(aoff) = clamp(0, args(aoff), 1);
            break;
        case 3:
            args(aoff + 0) = clamp(0, args(aoff + 0), 1);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1);
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

}

#endif /* POLYQUAD_SHAPES_QUAD_HPP */
