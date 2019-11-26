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

#ifndef POLYQUAD_SHAPES_TRI_HPP
#define POLYQUAD_SHAPES_TRI_HPP

#include "shapes/base.hpp"
#include "utils/ortho_poly.hpp"

#include <Eigen/Dense>

#include <cassert>

namespace polyquad {

template<typename T>
class TriDomain : public BaseDomain<TriDomain<T>, T, 2, 3>
{
public:
    typedef BaseDomain<TriDomain<T>, T, 2, 3> Base;
    typedef typename Base::MatrixXT MatrixXT;
    typedef typename Base::VectorXT VectorXT;
    typedef typename Base::ArrayXT ArrayXT;
    typedef typename Base::MatrixPtsT MatrixPtsT;
    typedef typename Base::VectorOrb VectorOrb;

    typedef Eigen::Matrix<T, 2, 1> Vector2T;

public:
    TriDomain() : Base(sqrt(T(2)))
    {}

    static bool validate_orbit(const VectorOrb& orb)
    { return orb(0) <= 1; }

private:
    friend class BaseDomain<TriDomain<T>, T, 2, 3>;

    static constexpr int npts_for_orbit[] = {1, 3, 6};
    static constexpr int narg_for_orbit[] = {0, 1, 2};

    constexpr int nbfn_for_qdeg(int qdeg) const;

    void expand_orbit(int i, int aoff, int poff,
                      const VectorXT& args, MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pq, D2 out) const;

    void clamp_arg(int i, int aoff, VectorXT& args) const;

    void sort_arg(int i, int aoff, VectorXT& args) const;

private:
    Vector2T bary_to_cart(const T& p1, const T& p2, const T& p3) const
    { return {-p1 + p2 - p3, -p1 - p2 + p3}; }
};

template<typename T>
inline constexpr int
TriDomain<T>::nbfn_for_qdeg(int qdeg) const
{
    int n = 0;

    for (int i = 0; i <= qdeg; i += 2)
        for (int j = i; j <= qdeg - i; ++j, ++n);

    return n;
}

template<typename T>
EIGEN_ALWAYS_INLINE void
TriDomain<T>::expand_orbit(int i, int aoff, int poff,
                           const VectorXT& args, MatrixPtsT& pts) const
{
    switch (i)
    {
        case 0:
        {
            const T& a = static_cast<T>(1) / 3;
            pts.row(poff) = bary_to_cart(a, a, a);
            break;
        }
        case 1:
        {
            const T& a = args(aoff);
            pts.row(poff + 0) = bary_to_cart(a, a, 1 - 2*a);
            pts.row(poff + 1) = bary_to_cart(a, 1 - 2*a, a);
            pts.row(poff + 2) = bary_to_cart(1 - 2*a, a, a);
            break;
        }
        case 2:
        {
            const T& a = args(aoff + 0);
            const T& b = args(aoff + 1);
            pts.row(poff + 0) = bary_to_cart(a, b, 1 - a - b);
            pts.row(poff + 1) = bary_to_cart(a, 1 - a - b, b);
            pts.row(poff + 2) = bary_to_cart(b, a, 1 - a - b);
            pts.row(poff + 3) = bary_to_cart(b, 1 - a - b, a);
            pts.row(poff + 4) = bary_to_cart(1 - a - b, a, b);
            pts.row(poff + 5) = bary_to_cart(1 - a - b, b, a);
            break;
        }
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline void
TriDomain<T>::seed_orbit(int i, int aoff, VectorXT& args)
{
    const std::array hist1a{240, 143, 119, 82, 117, 81, 58, 122, 143, 236};
    const std::array hist2a{929, 505, 148, 148, 192, 43, 39, 29, 3, 1};
    const std::array hist2b{123, 218, 258, 202, 265, 339, 197, 203, 222, 10};

    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = this->rand(0.0, 0.5, hist1a);
            break;
        case 2:
            args(aoff + 0) = this->rand(0, 1.0 / 3.0, hist2a);
            args(aoff + 1) = this->rand(0.0, 0.5, hist2b);
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
template<typename D1, typename D2>
inline void
TriDomain<T>::eval_orthob_block(const D1 pq, D2 out) const
{
    typedef Eigen::Array<T, D1::RowsAtCompileTime, 1> ArrayT;

    const auto& p = pq.col(0);
    const auto& q = pq.col(1);

    const ArrayT a = (q != 1).select(2*(1 + p)/(1 - q) - 1, 0);
    const ArrayT b = q;

    T pow2ip1 = 0.5;

    ArrayT pow1mbi = ArrayT::Constant(p.size(), 1);

    EvenLegendreP<ArrayT> jpa(a);

    for (int i = 0, off = 0; i <= this->qdeg(); i += 2)
    {
        JacobiP<ArrayT> jpb(2*i + 1, 0, b);

        for (int j = i; j <= this->qdeg() - i; ++j, ++off)
        {
            T cij = sqrt(T((2*i + 1)*(2*i + 2*j + 2)))*pow2ip1;

            out.row(off) = cij*pow1mbi*jpa(i)*jpb(j);
        }

        pow1mbi *= (1 - b)*(1 - b);
        pow2ip1 /= 4;
    }
}

template<typename T>
inline void
TriDomain<T>::clamp_arg(int i, int aoff, VectorXT& args) const
{
    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = clamp(0, args(aoff), 0.5);
            break;
        case 2:
            args(aoff + 0) = clamp(0, args(aoff + 0), 1);
            args(aoff + 1) = clamp(0, args(aoff + 1), 1 - args(aoff + 0));
            break;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline void
TriDomain<T>::sort_arg(int i, int aoff, VectorXT& args) const
{
    if (i == 2)
    {
        T baryc[] =
        {
            args(aoff + 0),
            args(aoff + 1),
            1 - args(aoff + 0) - args(aoff + 1)
        };
        std::sort(baryc, baryc + 3);
        std::copy(baryc, baryc + 2, args.data() + aoff);
    }
}

}

#endif /* POLYQUAD_SHAPES_TRI_HPP */
