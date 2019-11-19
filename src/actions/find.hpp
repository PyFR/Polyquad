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

#ifndef POLYQUAD_ACTIONS_FIND_HPP
#define POLYQUAD_ACTIONS_FIND_HPP

#include "config.h"

#include "utils/io.hpp"
#ifdef POLYQUAD_HAVE_MPI
# include "utils/serialization.hpp"
#endif
#include "utils/timer.hpp"

#ifdef POLYQUAD_HAVE_MPI
# include <boost/mpi.hpp>
#endif
#include <boost/program_options.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <vector>

namespace polyquad {

namespace mpi = boost::mpi;

template<typename VT, typename T>
bool
seen_rule(const std::vector<VT>& rules, const VT& newr, const T& tol)
{
    for (const auto& ri : rules)
        if ((newr - ri).squaredNorm() < tol)
            return true;

    return false;
}

template<template<typename> class Domain, typename T>
void
process_find(const boost::program_options::variables_map& vm)
{
    typedef typename Domain<T>::MatrixXT MatrixXT;
    typedef typename Domain<T>::VectorXT VectorXT;
    typedef typename Domain<T>::MatrixPtsT MatrixPtsT;
    typedef typename Domain<T>::VectorOrb VectorOrb;

#ifdef POLYQUAD_HAVE_MPI
    mpi::environment env;
    mpi::communicator world;

    const int rank = world.rank();
#else
    const int rank = 0;
#endif

    // Pinciple domain
    Domain<T> dom;

    // Flags
    const bool poswts = vm["positive"].as<bool>();
    const bool verbose = vm["verbose"].as<bool>();
    const bool twophase = vm["two-phase"].as<bool>();

    // Desired quadrature degree and point count
    const int qdeg = vm["qdeg"].as<int>();
    const int npts = vm["npts"].as<int>();
    const int maxfev = vm["maxfev"].as<int>();

    const int nprelim = twophase ? vm["nprelim"].as<int>() : 0;
    double runtime = vm["walltime"].as<int>();

    // Floating point tolerance and output precision
    const T tol = vm.count("tol") ? static_cast<T>(vm["tol"].as<double>())
                                  : Eigen::NumTraits<T>::dummy_precision();
    const int outprec = vm["output-prec"].as<int>();

    T norm;
    VectorXT args;

    // Decompose npts into symmetric orbital configurations
    auto orbits = dom.symm_decomps(npts);

    // Sequence in which the orbits should be considered
    std::vector<int> orbitseq(orbits.size());
    std::iota(std::begin(orbitseq), std::end(orbitseq), 0);

    if (twophase)
    {
        // Decide how many orbits to consider
        const int tryorbs = std::min<int>(std::max<int>(5, orbits.size() / 10),
                                          orbits.size());

        if (rank == 0 && verbose)
            std::cerr << "Phase I: Decomposition selection" << std::endl;

        Timer t;
        std::vector<double> norms(orbits.size(), 1000);

        for (int i = 0; i < orbits.size(); ++i)
        {
            dom.configure(qdeg, orbits[i]);

            for (int j = 0; j < nprelim; ++j)
            {
                // Seed the orbit
                dom.seed();

                // Attempt to minimise
                std::tie(norm, args) = dom.minimise(maxfev);

                // Save the norm
                norms[i] = std::min(static_cast<double>(norm), norms[i]);
            };
        }

#ifdef POLYQUAD_HAVE_MPI
        mpi::all_reduce(world, mpi::inplace(&norms.front()),
                        norms.size(), mpi::minimum<double>());
#endif

        // Determine the order in which we should examine the orbits
        std::sort(std::begin(orbitseq), std::end(orbitseq),
                  [&norms](int i, int j) { return norms[i] < norms[j]; });

        orbitseq.resize(tryorbs);

        if (rank == 0 && verbose)
            std::cerr << "Phase II: Proceeding with top " << tryorbs
                      << " decompositions" << std::endl;

        // Account for the time spent in decomposition selection
        runtime -= t.elapsed();
    }

    // Discovered rules for each configuration
    std::vector<std::vector<VectorXT>> rules(orbits.size());

    for (int j = 0; j < orbitseq.size(); ++j)
    {
        const int i = orbitseq[j];

        // Configure the domain for this orbit
        dom.configure(qdeg, orbits[i]);

        if (verbose && rank == 0)
            std::cerr << "Decomposition " << (i + 1) << "/"
                      << orbits.size() << " (" << dom.ndof() << " DOF): "
                      << orbits[i].transpose() << std::endl;

        Timer t;
        do
        {
            // Seed the orbit
            dom.seed();

            // Attempt to minimise
            std::tie(norm, args) = dom.minimise(maxfev);

            // See if the minimisation was successful
            if (norm < tol
             && (!poswts || (dom.wts(args).minCoeff() > 0))
             && !seen_rule(rules[i], args, tol))
            {
                rules[i].push_back(args);

                if (verbose)
                    std::cerr << '.' << std::flush;
            }
        } while (orbitseq.size()*t.elapsed() < runtime
              && rules[i].size() < 1000);

#ifdef POLYQUAD_HAVE_MPI
        if (rank == 0)
        {
            std::vector<std::vector<VectorXT>> gr;
            mpi::gather(world, rules[i], gr, 0);

            // Merge the rules from other ranks into our vector
            for (auto it = std::cbegin(gr) + 1; it != std::cend(gr); ++it)
                for (const auto& r : *it)
                    if (!seen_rule(rules[i], r, tol))
                        rules[i].push_back(r);
        }
        else
        {
            mpi::gather(world, rules[i], 0);
            rules[i].clear();
        }
#endif

        if (rank == 0)
        {
            if (verbose)
            {
                size_t nr = rules[i].size();

                if (nr)
                    std::cerr << '\n';
                std::cerr << "Rule count: " << nr << std::endl;
            }

            for (const auto& r : rules[i])
            {
                std::cout << "# Rule degree: " << qdeg
                          << " ("  << npts << " pts)\n";
                print_compact(std::cout, outprec, orbits[i], r);
                std::cout << std::endl;
            }
        }
    }
}

}

#endif /* POLYQUAD_ACTIONS_FIND_HPP */
