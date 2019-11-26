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

#ifndef POLYQUAD_ACTIONS_EVAL_HPP
#define POLYQUAD_ACTIONS_EVAL_HPP

#include "utils/io.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/program_options.hpp>
#include <Eigen/Dense>

#include <tuple>
#include <iostream>

namespace polyquad {

template<template<typename> class Domain, typename T>
void
process_eval(const boost::program_options::variables_map& vm)
{
    typedef typename Domain<T>::MatrixXT MatrixXT;
    typedef typename Domain<T>::VectorXT VectorXT;
    typedef typename Domain<T>::VectorOrb VectorOrb;

    // Pinciple domain
    Domain<T> dom;

    // Quadrature degree of the rule and the max number of fn evals
    const int qdeg = vm["qdeg"].as<int>();
    const int maxfev = vm["maxfev"].as<int>();

    // Floating point tolerance and output precision
    const T tol = vm.count("tol") ? static_cast<T>(vm["tol"].as<double>())
                                  : Eigen::NumTraits<T>::dummy_precision();
    const int outprec = vm["output-prec"].as<int>();

    // Flags
    const bool refine = vm["refine"].as<bool>();
    const bool verbose = vm["verbose"].as<bool>();

    // Input stream
    boost::iostreams::filtering_istream ifs;
    ifs.push(comment_filter());
    ifs.push(std::cin);
    ifs.exceptions(std::ios_base::failbit |
                   std::ios_base::badbit  |
                   std::ios_base::eofbit);

    try
    {
        while (true)
        {
            // Read in the orbital decomposition
            VectorOrb orb;
            ifs >> orb;

            // Configure the domain
            dom.configure(qdeg, false, orb);

            // Now read in the arguments (i.e, the rule)
            VectorXT args = VectorXT::Zero(dom.ndof());
            ifs >> args;

            // Compute the residual norm
            VectorXT resid(dom.nbfn());
            dom.wts(args, &resid);
            T norm = resid.norm();

            if (verbose)
                std::cerr << resid;

            std::cout << "# Rule degree: " << qdeg << "\n"
                      << "# Residual norm: " << norm << "\n";

            if (refine)
            {
                // Take the current rule to be the seed for minimisation
                dom.seed(args);

                std::tie(norm, args) = dom.minimise(maxfev);

                std::cout << "# Refined residual norm: " << norm << "\n";
            }

            print_compact(std::cout, outprec, orb, args);

            if (verbose)
                std::cerr << std::endl;
        }
    }
    catch (std::ios_base::failure& fail)
    {
        if (!ifs.eof())
            throw std::runtime_error("Malformed rule");
    }
}

}

#endif /* POLYQUAD_ACTIONS_EVAL_HPP */
