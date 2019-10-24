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

#ifndef POLYQUAD_SHAPES_BASE_HPP
#define POLYQUAD_SHAPES_BASE_HPP

#include <Eigen/Dense>
#include <Eigen/LevenbergMarquardt>

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

namespace polyquad {

template<typename T1, typename T2, typename T3>
T2 clamp(const T1& l, const T2& v, const T3& h)
{
    return (v < l) ? l : (v > h) ? h : v;
}

template<typename Derived, typename T, int Ndim, int Norbits>
class BaseDomain
{
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;
    typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayXT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Ndim> MatrixPtsT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixObatT;
    typedef Eigen::Matrix<int, Norbits, 1> VectorOrb;

public:
    BaseDomain(const T& f0)
        : f0_(f0)
        , rand_eng_(std::random_device()())
    {}

    void configure(int qdeg, const VectorOrb& orbits);

    std::vector<VectorOrb> symm_decomps(int npts) const;

    void seed();

    void seed(const VectorXT& args)
    { args_ = args; }

    VectorXT clamp_args(const VectorXT& args) const;

    void expand(const VectorXT& args, MatrixPtsT& pts) const;

    void eval_orthob(const MatrixPtsT& pts, MatrixObatT& out) const;

    void expand_wts(const VectorXT& wargs, VectorXT& wts) const;

    std::tuple<T, VectorXT> minimise(int maxfev);

    const VectorXT& wts(const VectorXT& args,
                        VectorXT* resid=nullptr);

    int qdeg() const
    { return qdeg_; }

    int ndim() const
    { return Ndim; }

    int nbfn() const
    { return static_cast<const Derived&>(*this).nbfn_for_qdeg(qdeg_); }

    const VectorOrb& orbits() const
    { return orbits_; }

    int npts() const;

    int nwts() const
    { return orbits_.sum(); }

    int ndof() const;

protected:
    double rand(double a, double b)
    { return std::uniform_real_distribution<double>(a, b)(rand_eng_); }

private:
    void symm_decomps_recurse(VectorOrb coeffs,
                              int sum,
                              VectorOrb& partsoln,
                              std::vector<VectorOrb>& solns) const;

    int qdeg_;
    const T f0_;

    VectorOrb orbits_;
    VectorXT args_;

    std::mt19937 rand_eng_;

    mutable MatrixPtsT pts_;
    mutable MatrixObatT obat_;
    mutable Eigen::HouseholderQR<MatrixXT> qr_;
    mutable MatrixXT A_;
    mutable VectorXT b_;
    mutable VectorXT wts_;
};

template<typename Derived, typename T, int Ndim, int Norbits>
inline void
BaseDomain<Derived, T, Ndim, Norbits>::configure(
        int qdeg,
        const VectorOrb& orbits)
{
    using Eigen::HouseholderQR;

    qdeg_ = qdeg;
    orbits_ = orbits;

    // Pre-allocate some scratch space
    pts_.resize(npts(), Ndim);
    obat_.resize(nbfn(), npts());
    A_.resize(nbfn(), nwts());
    b_.resize(nbfn());
    wts_.resize(nwts());
    qr_ = HouseholderQR<MatrixXT>(A_.rows(), A_.cols());

    b_.fill(0);
    b_(0) = f0_;
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline std::vector<typename BaseDomain<Derived, T, Ndim, Norbits>::VectorOrb>
BaseDomain<Derived, T, Ndim, Norbits>::symm_decomps(int npts) const
{
    const Derived& derived = static_cast<const Derived&>(*this);
    VectorOrb coeffs;

    for (int i = 0; i < Norbits; ++i)
        coeffs(i) = derived.npts_for_orbit(i);

    std::vector<VectorOrb> solns;
    VectorOrb partsoln = VectorOrb::Zero();

    symm_decomps_recurse(coeffs, npts, partsoln, solns);

    return solns;
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline void
BaseDomain<Derived, T, Ndim, Norbits>::seed()
{
    Derived& derived = static_cast<Derived&>(*this);

    args_ = VectorXT::Zero(ndof());

    int aoff = 0;

    for (int i = 0; i < Norbits; ++i)
    {
        int ainc = derived.narg_for_orbit(i);

        for (int j = 0; j < orbits_(i); ++j, aoff += ainc)
            derived.seed_orbit(i, aoff, args_);
    }
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline typename BaseDomain<Derived, T, Ndim, Norbits>::VectorXT
BaseDomain<Derived, T, Ndim, Norbits>::clamp_args(const VectorXT& args) const
{
    const Derived& derived = static_cast<const Derived&>(*this);
    VectorXT nargs = args;

    for (int i = 0, aoff = 0; i < orbits_.rows(); ++i)
    {
        int ainc = derived.narg_for_orbit(i);

        for (int j = 0; j < orbits_(i); ++j, aoff += ainc)
            derived.clamp_arg(i, aoff, nargs);
    }

    return nargs;
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline void
BaseDomain<Derived, T, Ndim, Norbits>::expand(
        const VectorXT& args,
        MatrixPtsT& pts) const
{
    const Derived& derived = static_cast<const Derived&>(*this);
    int aoff = 0, poff = 0;

    for (int i = 0; i < Norbits; ++i)
    {
        int ainc = derived.narg_for_orbit(i);
        int pinc = derived.npts_for_orbit(i);

        for (int j = 0; j < orbits_(i); ++j, aoff += ainc, poff += pinc)
            derived.expand_orbit(i, aoff, poff, args, pts);
    }
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline void
BaseDomain<Derived, T, Ndim, Norbits>::eval_orthob(
        const MatrixPtsT& pts,
        MatrixObatT& out) const
{
    const Derived& derived = static_cast<const Derived&>(*this);

    int n = out.cols();

    if (std::is_fundamental<T>::value)
    {
        int n16 = (n / 16)*16, n8 = (n / 8)*8, n4 = (n / 4)*4, n2 = (n / 2)*2;

        // Process points in blocks of sixteen
        for (int i = 0; i < n16; i += 16)
            derived.eval_orthob_block(pts.array().template middleRows<16>(i),
                                      out.array().template middleCols<16>(i));

        // Cleanup
        if (n16 < n8)
            derived.eval_orthob_block(pts.array().template middleRows<8>(n16),
                                      out.array().template middleCols<8>(n16));
        if (n8 < n4)
            derived.eval_orthob_block(pts.array().template middleRows<4>(n8),
                                      out.array().template middleCols<4>(n8));
        if (n4 < n2)
            derived.eval_orthob_block(pts.array().template middleRows<2>(n4),
                                      out.array().template middleCols<2>(n4));
        if (n2 < n)
            derived.eval_orthob_block(pts.array().template middleRows<1>(n2),
                                      out.array().template middleCols<1>(n2));
    }
    else
    {
        derived.eval_orthob_block(pts.array(), out.array());
    }
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline void
BaseDomain<Derived, T, Ndim, Norbits>::expand_wts(
        const VectorXT& wargs,
        VectorXT& wtsout) const
{
    const Derived& derived = static_cast<const Derived&>(*this);

    for (int i = 0, poff = 0, woff = 0; i < Norbits; ++i)
    {
        int pinc = derived.npts_for_orbit(i);

        for (int j = 0; j < orbits_(i); ++j, poff += pinc, ++woff)
            wtsout.segment(poff, pinc).fill(wargs(woff));
    }
}


template<typename Derived, typename T, int Ndim, int Norbits>
inline std::tuple<T, typename BaseDomain<Derived, T, Ndim, Norbits>::VectorXT>
BaseDomain<Derived, T, Ndim, Norbits>::minimise(int maxfev)
{
    struct min_functor : Eigen::DenseFunctor<T>
    {
        min_functor(Derived& dom)
            : Eigen::DenseFunctor<T>(dom.ndof(), dom.nbfn() + 1)
            , dom_(dom)
        {}

        int operator()(const VectorXT& x, VectorXT& f) const
        {
            // Compute the residual
            dom_.wts(x, &f);

            // Account for invalid arguments
            f(f.size() - 1) = (x - dom_.clamp_args(x)).norm();

            return 0;
        }

        Derived& dom_;
    };

    Derived& derived = static_cast<Derived&>(*this);

    typedef Eigen::NumericalDiff<min_functor> min_functor_ndiff;
    min_functor f(derived);
    min_functor_ndiff fd(f);

    // Perform the minimisation
    Eigen::LevenbergMarquardt<min_functor_ndiff> lm(fd);
    lm.setMaxfev(maxfev > 0 ? maxfev : 40*qdeg_*qdeg_);
    lm.minimize(args_);

    // Clamp the arguments to ensure all points are inside the domain
    args_ = derived.clamp_args(args_);

    // Compute the residual of these clamped points
    VectorXT resid(derived.nbfn());
    derived.wts(args_, &resid);

    // Return
    return std::make_tuple(resid.norm(), args_);
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline const typename BaseDomain<Derived, T, Ndim, Norbits>::VectorXT&
BaseDomain<Derived, T, Ndim, Norbits>::wts(
        const VectorXT& args, VectorXT* resid)
{
    const Derived& derived = static_cast<const Derived&>(*this);

    derived.expand(args, pts_);
    derived.eval_orthob(pts_, obat_);

    for (int i = 0, poff = 0, coff = 0; i < Norbits; ++i)
    {
        int pinc = derived.npts_for_orbit(i);

        for (int j = 0; j < orbits_(i); ++j, ++coff, poff += pinc)
            for (int l = 0; l < A_.rows(); ++l)
                A_(l, coff) = obat_.block(l, poff, 1, pinc).sum();
    }

    // Compute the optimal set of weights
    wts_ = qr_.compute(A_).solve(b_);

    if (resid)
    {
        (*resid).head(b_.size()).noalias() = A_*wts_;
        (*resid)(0) -= f0_;
    }

    return wts_;
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline int
BaseDomain<Derived, T, Ndim, Norbits>::npts() const
{
    const Derived& derived = static_cast<const Derived&>(*this);
    int s = 0;

    for (int i = 0; i < Norbits; ++i)
        s += orbits_(i)*derived.npts_for_orbit(i);

    return s;
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline int
BaseDomain<Derived, T, Ndim, Norbits>::ndof() const
{
    const Derived& derived = static_cast<const Derived&>(*this);
    int s = 0;

    for (int i = 0; i < Norbits; ++i)
        s += orbits_(i)*derived.narg_for_orbit(i);

    return s;
}

template<typename Derived, typename T, int Ndim, int Norbits>
inline void
BaseDomain<Derived, T, Ndim, Norbits>::symm_decomps_recurse(
        VectorOrb coeffs,
        int sum,
        VectorOrb& partsoln,
        std::vector<VectorOrb>& solns) const
{
    int index;
    int mcoeff = coeffs.maxCoeff(&index);
    int range = sum / mcoeff;

    if (range*mcoeff == sum)
    {
        partsoln(index) = range--;

        if (static_cast<const Derived&>(*this).validate_orbit(partsoln))
            solns.push_back(partsoln);
    }

    if (coeffs.count() == 1)
        return;

    while (range >= 0)
    {
        int rem = sum - range*mcoeff;

        VectorOrb coeffCopy = coeffs;
        coeffCopy(index) = 0;

        partsoln(index) = range--;

        for (int i = 0; i < coeffCopy.rows(); ++i)
            if (coeffCopy(i) != 0)
                partsoln(i) = 0;

        symm_decomps_recurse(coeffCopy, rem, partsoln, solns);
    }
}

}

#endif /* POLYQUAD_SHAPES_BASE_HPP */
