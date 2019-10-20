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

#ifndef POLYQUAD_UTILS_SERIALIZATION_HPP
#define POLYQUAD_UTILS_SERIALIZATION_HPP

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#ifdef POLYQUAD_HAVE_MPREAL
# include <mpreal.h>
#endif

#include <Eigen/Dense>

#include <string>

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

#endif /* POLYQUAD_UTILS_SERIALIZATION_HPP */
