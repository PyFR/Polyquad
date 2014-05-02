
#ifndef POLYQUAD_UTIL_HPP
#define POLYQUAD_UTIL_HPP

#include <boost/chrono.hpp>
#include <boost/iostreams/filter/line.hpp>

#ifdef POLYQUAD_HAVE_MPI
# include <boost/serialization/array.hpp>
# ifdef POLYQUAD_HAVE_MPREAL
#  include <mpreal.h>
# endif
#endif

#include <Eigen/Dense>

#include <istream>
#include <string>

#ifdef POLYQUAD_HAVE_MPI
namespace boost {
namespace serialization {

template<typename Archive, typename Scalar, int Rows, int Cols>
inline void
serialize(Archive& ar, Eigen::Matrix<Scalar, Rows, Cols>& m,
          unsigned int version)
{
    int rows, cols;

    if (Archive::is_saving::value)
    {
        rows = m.rows();
        cols = m.cols();
        ar & rows & cols & make_array(m.data(), m.size());
    }
    else
    {
        ar & rows & cols;
        m.resize(rows, cols);
        ar & make_array(m.data(), m.size());
    }
}

#ifdef POLYQUAD_HAVE_MPREAL
template<typename Archive>
inline void
serialize(Archive& ar, mpfr::mpreal& m, unsigned int version)
{
    if (Archive::is_saving::value)
    {
        const std::string& ms = m.toString();
        ar & const_cast<std::string&>(ms);
    }
    else
    {
        std::string ms;
        ar & ms;
        m = ms;
    }
}
#endif /* POLYQUAD_HAVE_MPREAL */

} }
#endif /* POLYQUAD_HAVE_MPI */

namespace polyquad {

class Timer
{
public:
    Timer() : start_(now())
    {}

    double elapsed() const
    { return now() - start_; }

private:
    static double now();

    const double start_;
};

inline double
Timer::now()
{
    using namespace boost::chrono;

    auto t = high_resolution_clock::now().time_since_epoch();
    auto d = duration_cast<duration<double>>(t);

    return d.count();
}


template<typename Derived>
std::istream&
operator>>(std::istream& in, Eigen::MatrixBase<Derived>& m)
{
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            in >> m(i, j);

    return in;
}

class comment_filter : public boost::iostreams::line_filter
{
    std::string do_filter(const std::string& line)
    { return (line.empty() || line[0] == '#') ? "" : line; }
};

template<typename T1, typename T2, typename T3>
T2 clamp(const T1& l, const T2& v, const T3& h)
{
    return (v < l) ? l : (v > h) ? h : v;
}


}

#endif /* POLYQUAD_UTIL_HPP */
