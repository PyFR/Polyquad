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

#ifndef POLYQUAD_UTILS_IO_HPP
#define POLYQUAD_UTILS_IO_HPP

#include <boost/iostreams/filter/line.hpp>
#include <Eigen/Dense>

#include <istream>
#include <string>

namespace polyquad {

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

template<typename D, typename R>
void
print_compact(std::ostream& out, int outprec, const D& decomp, const R& args)
{
    const Eigen::IOFormat fmt(outprec, Eigen::DontAlignCols);

    out << decomp.transpose().format(fmt) << "\n";
    out << args.format(fmt) << "\n";
}

}

#endif /* POLYQUAD_UTILS_IO_HPP */
