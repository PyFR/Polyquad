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

#include <boost/mpi/datatype.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

#include <Eigen/Dense>

#include <string>
#include <tuple>

namespace boost::serialization {

template<int N>
struct TupleSerialize
{
    template<class Archive, typename... Args>
    static void serialize(Archive& ar, std::tuple<Args...>& t)
    {
        ar & std::get<N - 1>(t);
        TupleSerialize<N - 1>::serialize(ar, t);
    }
};

template<>
struct TupleSerialize<0>
{
    template<class Archive, typename... Args>
    static void serialize(Archive&, std::tuple<Args...>&) {}
};

template<class Archive, typename... Args>
inline void
serialize(Archive& ar, std::tuple<Args...>& t, unsigned int)
{
    TupleSerialize<sizeof...(Args)>::serialize(ar, t);
}

template<typename Archive, typename Scalar, int Rows, int Cols>
inline void
serialize(Archive& ar, Eigen::Matrix<Scalar, Rows, Cols>& m, unsigned int)
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

}

namespace boost::mpi {

template<typename U>
struct is_mpi_datatype<std::tuple<U>> : public is_mpi_datatype<U>
{};

template<typename U, typename... V>
struct is_mpi_datatype<std::tuple<U, V...>>
    : public mpl::and_<is_mpi_datatype<U>,
                       is_mpi_datatype<std::tuple<V...>>>
{};

}

#endif /* POLYQUAD_UTILS_SERIALIZATION_HPP */
