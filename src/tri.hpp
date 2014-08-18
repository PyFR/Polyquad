
#ifndef POLYQUAD_TRI_HPP_
#define POLYQUAD_TRI_HPP_

#include "base.hpp"
#include "jacobi_poly.hpp"
#include "util.hpp"

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

    bool validate_orbit(const VectorOrb& orb) const
    { return orb(0) <= 1; }

private:
    friend class BaseDomain<TriDomain<T>, T, 2, 3>;

    int npts_for_orbit(int i) const;

    int narg_for_orbit(int i) const;

    int nbfn_for_qdeg(int qdeg) const;

    void expand_orbit(int i, int aoff, int poff,
                      const VectorXT& args, MatrixPtsT& pts) const;

    void seed_orbit(int i, int aoff, VectorXT& args);

    template<typename D1, typename D2>
    void eval_orthob_block(const D1 pq, D2 out) const;

    void clamp_arg(int i, int aoff, VectorXT& args) const;

private:
    Vector2T bary_to_cart(const T& p1, const T& p2, const T& p3) const
    { return {-p1 + p2 - p3, -p1 - p2 + p3}; }
};

template<typename T>
inline int
TriDomain<T>::npts_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 1;
        case 1:
            return 3;
        case 2:
            return 6;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline int
TriDomain<T>::narg_for_orbit(int i) const
{
    switch (i)
    {
        case 0:
            return 0;
        case 1:
            return 1;
        case 2:
            return 2;
        default:
            assert(0 && "Bad orbit"), abort();
    }
}

template<typename T>
inline int
TriDomain<T>::nbfn_for_qdeg(int qdeg) const
{
    int n = 0;

    for (int i = 0; i <= qdeg; ++i)
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
    switch (i)
    {
        case 0:
            break;
        case 1:
            args(aoff) = this->rand(0.0, 0.5);
            break;
        case 2:
            args(aoff + 0) = this->rand(0.0, 1.0 / 3.0);
            args(aoff + 1) = this->rand(0.0, 1.0 / 3.0);
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

    const auto& a = (q != 1).select(2*(1 + p)/(1 - q) - 1, 0);
    const auto& b = q;

    ArrayT pow1mbi = ArrayT::Constant(p.size(), 1);
    T pow2ip1 = 0.5;

    for (int i = 0, off = 0; i <= this->qdeg(); ++i)
    {
        for (int j = i; j <= this->qdeg() - i; ++j, ++off)
        {
            T cij = sqrt(T((2*i + 1)*(2*i + 2*j + 2)))*pow2ip1;

            out.row(off) = cij*pow1mbi
                         * jacobi_poly(i, 0, 0, a)
                         * jacobi_poly(j, 2*i + 1, 0, b);
        }

        pow1mbi *= 1 - b;
        pow2ip1 /= 2;
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

}

#endif /* POLYQUAD_TRI_HPP_ */
