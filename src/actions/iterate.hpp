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

#ifndef POLYQUAD_ACTIONS_ITERATE_HPP
#define POLYQUAD_ACTIONS_ITERATE_HPP

#include "config.h"

#include "utils/io.hpp"
#ifdef POLYQUAD_HAVE_MPI
# include "utils/serialization.hpp"
#endif
#include "utils/timer.hpp"

#ifdef POLYQUAD_HAVE_MPI
# include <boost/mpi.hpp>
# include <boost/serialization/utility.hpp>
#endif
#include <boost/program_options.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

namespace polyquad {

namespace mpi = boost::mpi;

#ifdef POLYQUAD_HAVE_MPI
static const int npts_tag = 1;
static const int rule_tag = 2;

static void
post_npts(mpi::communicator& world, int npts)
{
    const int rank = world.rank();
    const int size = world.size();

    std::vector<mpi::request> reqs;

    // Inform the other ranks about our discovery
    for (int i = 0; i < size; ++i)
        if (i != rank)
            reqs.emplace_back(world.isend(i, npts_tag, npts));

    mpi::wait_all(std::begin(reqs), std::end(reqs));
}

static void
probe_npts(mpi::communicator& world, int& ub)
{
    // See if any other ranks have improved on ub
    while (world.iprobe(mpi::any_source, npts_tag))
    {
        int i;
        world.recv(mpi::any_source, npts_tag, i);

        if (i < ub)
            ub = i;
    }
}

static void
post_rule(mpi::communicator& world, int npts, const std::string& rstr)
{
    // Broadcast npts
    post_npts(world, npts);

    // Send the rule to the root rank for printing
    if (world.rank() != 0)
        world.send(0, rule_tag, make_pair(npts, rstr));
}

static void
probe_rules(mpi::communicator& world, int& printed_npts)
{
    // Check if any rules have been forwarded to us
    while (world.iprobe(mpi::any_source, rule_tag))
    {
        std::pair<int, std::string> rule;
        world.recv(mpi::any_source, rule_tag, rule);

        if (rule.first < printed_npts)
        {
            std::cout << rule.second << std::flush;
            printed_npts = rule.first;
        }
    }
}
#endif

template<typename Generator>
int pick_npts(Generator& g, double lambda, int lb, int ub)
{
    int npts;
    std::exponential_distribution<double> dist(lambda);

    do
    {
        npts = ub - int(dist(g)) - 1;
    } while (npts <= lb);

    return npts;
}

template<typename D, typename R>
std::string
rule_to_str(int qdeg, int npts, const D& decomp, const R& args, int outprec)
{
    std::stringstream ss;

    ss << "# Rule degree: " << qdeg << " (" << npts << " pts)\n";
    print_compact(ss, outprec, decomp, args);
    ss << std::endl;

    return ss.str();
}

template<template<typename> class Domain, typename T>
void
process_iterate(const boost::program_options::variables_map& vm)
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

    // Min/max point count
    const int lb = vm["lb"].as<int>();
    int ub = vm["ub"].as<int>();

    // Desired quadrature degree
    const int qdeg = vm["qdeg"].as<int>();
    const int maxfev = vm["maxfev"].as<int>();

    const int nprelim = 5;
    const double lambda = 0.1;

    // Floating point tolerance and output precision
    const T tol = vm.count("tol") ? static_cast<T>(vm["tol"].as<double>())
                                  : Eigen::NumTraits<T>::dummy_precision();
    const int outprec = vm["output-prec"].as<int>();

    // Number of points in the last rule we printed
    int printed_npts = ub;

    // Random number generator
    std::mt19937 rand_eng((std::random_device()()));

    // Determine which point counts yield valid decompositions
    std::map<int, std::pair<std::vector<VectorOrb>, std::vector<double>>> nptsorbits;
    for (int i = lb + 1; i < ub; ++i)
    {
        std::vector<VectorOrb> orb = dom.symm_decomps(i);

        if (orb.size())
        {
            std::vector<double> norms(orb.size(), 1e8);

            nptsorbits.insert({i, make_pair(orb, norms)});
        }
    }

    int npts;
    T norm;
    VectorXT args;

    // Main search loop
start:
    // Prune nptsorbits
    nptsorbits.erase(nptsorbits.upper_bound(ub - 1), std::end(nptsorbits));

    // Return if no points remain
    if (!nptsorbits.size())
        return;

    // Pick npts
    do
    {
        npts = pick_npts(rand_eng, lambda, lb, ub);
    } while (!nptsorbits.count(npts));

    // Obtain the orbital decompositions of npts and their associated norms
    const auto& orbits = nptsorbits[npts].first;
    auto& norms = nptsorbits[npts].second;

    // Start the timer
    Timer t;

    // Update the norms associated with each orbit
    for (int i = 0; i < orbits.size(); ++i)
    {
        dom.configure(qdeg, orbits[i]);

        for (int j = 0; j < nprelim; ++j)
        {
            // Seed the orbit
            dom.seed();

            // Attempt to minimise
            std::tie(norm, args) = dom.minimise(maxfev);

            // Update the norm
            norms[i] = std::min(norm, norms[i]);

            // If we're lucky we may have found a rule
            if (norm < tol && (!poswts || (dom.wts(args).minCoeff() > 0)))
            {
                // Convert the rule to a string
                auto rstr = rule_to_str(qdeg, npts, orbits[i], args, outprec);

#ifdef POLYQUAD_HAVE_MPI
                post_rule(world, npts, rstr);
#endif

                if (rank == 0)
                    std::cout << rstr << std::flush;

                // Update ub and continue
                ub = npts;
                goto start;
            }
        };
    }

    // Use these norms to determine a sequence in which to examine the orbits
    std::vector<int> orbitseq(orbits.size());
    std::iota(std::begin(orbitseq), std::end(orbitseq), 0);
    std::sort(std::begin(orbitseq), std::end(orbitseq),
              [&norms](int i, int j) { return norms[i] < norms[j]; });

    // Limit ourselves to the top ten orbits
    const int tryorbs = std::min<int>(10, orbits.size());

    // Record how long we spent in the preliminary phase
    const double pretime = t.elapsed();

    // Reset the time for the main phase
    t.reset();

    for (int j = 0; ; j = (j + 1) % tryorbs)
    {
        const int i = orbitseq[j];

#ifdef POLYQUAD_HAVE_MPI
        // If we are the root rank see if any rules need printing
        if (rank == 0)
            probe_rules(world, printed_npts);

        // See if another rank has made progress
        probe_npts(world, ub);
        if (npts >= ub)
            goto start;
#endif

        // Configure the domain for this orbit
        dom.configure(qdeg, orbits[i]);

        // Seed the orbit
        dom.seed();

        // Attempt to minimise
        std::tie(norm, args) = dom.minimise(maxfev);

        // Update the norm for this orbit
        norms[i] = std::min(norm, norms[i]);

        // See if a rule has been found
        if (norm < tol && (!poswts || (dom.wts(args).minCoeff() > 0)))
        {
            // Convert the rule to a string
            auto rstr = rule_to_str(qdeg, npts, orbits[i], args, outprec);

#ifdef POLYQUAD_HAVE_MPI
            // Inform other ranks of our progress
            post_rule(world, npts, rstr);
#endif

            if (rank == 0)
            {
                std::cout << rstr << std::flush;
                printed_npts = npts;
            }

            // Update ub and continue
            ub = npts;
            goto start;
        }
        // If too much time has elapsed go back to the start
        else if (t.elapsed() > 100*pretime)
            goto start;
    }
}

}

#endif /* POLYQUAD_ACTIONS_ITERATE_HPP */
