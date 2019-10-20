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

#include "config.h"

#include "utils/io.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/program_options.hpp>
#include <Eigen/Dense>

#include <iostream>

namespace polyquad {

template<template<typename> class Domain, typename T>
void
process_expand(const boost::program_options::variables_map& vm)
{
    typedef typename Domain<T>::MatrixXT MatrixXT;
    typedef typename Domain<T>::VectorXT VectorXT;
    typedef typename Domain<T>::MatrixPtsT MatrixPtsT;
    typedef typename Domain<T>::VectorOrb VectorOrb;

    // Principal domain
    Domain<T> dom;

    // Quadrature degree of the rule and the max number of fn evals
    const int qdeg = vm["qdeg"].as<int>();

    // If to output a header or not
    bool header = vm["header"].as<bool>();

    // Input stream
    boost::iostreams::filtering_istream ifs;
    ifs.push(comment_filter());
    ifs.push(std::cin);
    ifs.exceptions(std::ios_base::failbit |
                   std::ios_base::badbit  |
                   std::ios_base::eofbit);

    // Output formatting
    const Eigen::IOFormat fmt(vm["output-prec"].as<int>(), 0, "  ");

    try
    {
        while (true)
        {
            // Read in the orbital decomposition
            VectorOrb orb;
            ifs >> orb;

            // Configure the domain
            dom.configure(qdeg, orb);

            // Now read in the arguments (i.e, the rule)
            VectorXT args = VectorXT::Zero(dom.ndof());
            ifs >> args;

            // Expand the points
            MatrixPtsT pts(dom.npts(), dom.ndim());
            dom.expand(args, pts);

            // Compute and expand the weights
            VectorXT wts(dom.npts());
            dom.expand_wts(dom.wts(args), wts);

            // Concatenate
            MatrixXT out(dom.npts(), dom.ndim() + 1);
            out << pts, wts;

            if (header)
                std::cout << "# Rule degree: " << qdeg << "\n";

            std::cout << out.format(fmt) << "\n" << std::endl;
        }
    }
    catch (std::ios_base::failure& fail)
    {
        if (!ifs.eof())
            throw std::runtime_error("Malformed rule");
    }
}

}
