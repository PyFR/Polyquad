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

#ifndef POLYQUAD_SHAPES_TET_HPP
#define POLYQUAD_SHAPES_TET_HPP

#include "shapes/base.hpp"
#include "utils/ortho_poly.hpp"

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

    static bool validate_orbit(const VectorOrb& orb)
    { return orb(0) <= 1; }

private:
    friend class BaseDomain<TetDomain<T>, T, 3, 5>;

    static constexpr int npts_for_orbit[] = {1, 4, 6, 12, 24};
    static constexpr int narg_for_orbit[] = {0, 1, 1,  2,  3};
    static constexpr int nbfn_for_qdeg(int qdeg);

    void expand_orbit(int i, int aoff, int poff, const VectorXT& args,
                      MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pqr, D2 out) const;

    template<typename ReplaceF>
    static void collapse_arg(int i, int aoff, const VectorXT& args, ReplaceF replace, const T& tol);

    static void clamp_arg(int i, int aoff, VectorXT& args);

    static void sort_arg(int i, int aoff, VectorXT& args);

private:
    Vector3T bary_to_cart(const T& p1, const T& p2, const T& p3,
                          const T& p4) const
    { return {-p4 - p3 + p2 - p1, -p4 + p3 - p2 - p1, p4 - p3 - p2 - p1}; }
};

template<typename T>
inline constexpr int
TetDomain<T>::nbfn_for_qdeg(int qdeg)
{
    int n = 0;

    for (int i = 0; i <= qdeg; i += 2)
        for (int j = i; j <= qdeg - i; ++j)
            for (int k = j; k <= qdeg - i - j; ++k, ++n);

    return n;
}

template<typename T>
void
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
            args(aoff) = this->rand(0.0, 0.25);
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

    const ArrayT a = (q != -r).select(-2*(1 + p)/(q + r) - 1, 0);
    const ArrayT b = (r != 1).select(2*(1 + q)/(1 - r) - 1, 0);
    const auto& c = r;

    const T pow2m32 = exp2(T(-1.5));
    T pow2mi = 1;

    ArrayT pow1mbi = ArrayT::Constant(p.size(), 1);
    ArrayT pow1mci = ArrayT::Constant(p.size(), 1);

    EvenLegendreP<ArrayT> jpa(a);

    for (int i = 0, off = 0; i <= this->qdeg(); i += 2)
    {
        T ci = pow2mi*pow2mi*pow2m32;
        T pow2mj = pow2mi;
        ArrayT pow1mcj = pow1mci;
        JacobiP<ArrayT> jpb(2*i + 1, 0, b);

        for (int j = i; j <= this->qdeg() - i; ++j)
        {
            T cij = ci*pow2mj;
            JacobiP<ArrayT> jpc(2*(i + j + 1), 0, c);

            for (int k = j; k <= this->qdeg() - i - j; ++k, ++off)
            {
                T cijk = cij*sqrt(T((2*(k + j + i) + 3)*(i + j + 1)*(4*i + 2)));

                out.row(off) = cijk*pow1mbi*pow1mci*pow1mcj
                             * jpa(i)*jpb(j)*jpc(k);
            }

            pow2mj /= 2;
            pow1mcj *= 1 - c;
        }

        pow2mi /= 4;
        pow1mbi *= (1 - b)*(1 - b);
        pow1mci *= (1 - c)*(1 - c);
    }
}

template<typename T>
template<typename ReplaceF>
void inline
TetDomain<T>::collapse_arg(int i, int aoff, const VectorXT& args,
                           ReplaceF replace, const T& tol)
{
    const T half = T(1) / 2, fourth = T(1) / 4;

    if ((i == 1 || i == 2) && abs(args(aoff) - fourth) < tol)
        replace(0);
    else if (i == 3)
    {
        const T a = args(aoff + 0), b = args(aoff + 1);

        if (abs(a - fourth) < tol && abs(b - fourth) < tol)
            replace(0);
        else if (abs(a - b) < tol)
            replace(1, a);
        else if (abs(b - (half - a)) < tol)
            replace(2, a);
    }
    else if (i == 4)
    {
        const T a = args(aoff + 0), b = args(aoff + 1), c = args(aoff + 2);
        const T d = 1 - a - b - c;

        if (abs(d - fourth) < tol)
            replace(0);
        else if (abs(a - c) < tol)
            replace(1, a);
        else if (abs(b - d) < tol)
            replace(1, b);
        else if (abs(a - b) < tol && abs(c - d) < tol)
            replace(2, a);
        else if (abs(a - b) < tol)
            replace(3, a, c);
        else if (abs(b - c) < tol)
            replace(3, b, a);
        else if (abs(c - d) < tol)
            replace(3, c, a);
    }
}

template<typename T>
inline void
TetDomain<T>::clamp_arg(int i, int aoff, VectorXT& args)
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

template<typename T>
inline void
TetDomain<T>::sort_arg(int i, int aoff, VectorXT& args)
{
    if (i == 2)
        args(aoff) = std::min(args(aoff), 0.5 - args(aoff));
    else if (i == 3)
        args(aoff + 1) = std::min(args(aoff + 1), 1 - 2*args(aoff + 0) - args(aoff + 1));
    else if (i == 4)
    {
        T baryc[] =
        {
            args(aoff + 0),
            args(aoff + 1),
            args(aoff + 2),
            1 - args(aoff + 0) - args(aoff + 1) - args(aoff + 2)
        };
        std::sort(baryc, baryc + 4);
        std::copy(baryc, baryc + 3, args.data() + aoff);
    }
}

}

#endif /* POLYQUAD_SHAPES_TET_HPP */
