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

#include <any>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace polyquad {

#ifdef POLYQUAD_HAVE_MPI
namespace mpi = boost::mpi;
#endif
namespace po = boost::program_options;

template<typename D, typename R>
std::string
rule_to_str(int qdeg, int npts, const D& decomp, const R& args, int outprec)
{
    std::stringstream ss;

    ss << "# Rule degree: " << qdeg << " (" << npts << " pts)\n";
    print_compact(ss, outprec, decomp, args);
    ss << "\n";

    return ss.str();
}

template<template<typename> class Domain, typename T>
class IterateAction
{
public:
    typedef typename Domain<T>::MatrixXT MatrixXT;
    typedef typename Domain<T>::VectorXT VectorXT;
    typedef typename Domain<T>::MatrixPtsT MatrixPtsT;
    typedef typename Domain<T>::VectorOrb VectorOrb;
    typedef typename Domain<T>::VectorOrbArgs VectorOrbArgs;

    typedef std::tuple<std::pair<int, int>, int, double> Stats;

    struct DecompRecord
    {
        VectorOrb orbit;
        int ntries;
        double resid;
    };

    enum
    {
        RuleTag,
        NptsTag,
        StatsTag
    };

public:
    IterateAction(const po::variables_map& vm);

    void run();

private:
    static constexpr double InitResid = 1e8;

    void update_decomps(int ub);

    void pick_decomp();

    bool attempt_to_minimise(int maxfev, double* rresid=nullptr);

    void attempt_to_reduce(const VectorXT& args);

#ifdef POLYQUAD_HAVE_MPI
    void pump_messages();

    template<typename Object>
    void post_message(int tag, const Object& obj);
#endif

private:
    std::mt19937 rand_eng_;

    Domain<T> dom_;
    std::vector<VectorOrbArgs> seen_red_;

    Timer twall_;
    int runtime_;

    std::map<std::pair<int, int>, DecompRecord> drecords_;
    std::vector<std::pair<int, int>> dixs_;

    const int qdeg_;
    const int maxfev_;

    const bool poswts_;
    const bool verbose_;

    const int lb_;
    int ub_;

    const int ntries_;

    T tol_;
    const T collapse_tol_;
    const int outprec_;

    std::pair<int, int> active_;

#ifdef POLYQUAD_HAVE_MPI
    mpi::communicator world_;

    std::list<std::pair<std::any, std::vector<mpi::request>>> reqs_;

    int rank_;
    int size_;
    int printed_npts_;
#endif
};

template<template<typename> class Domain, typename T>
inline
IterateAction<Domain, T>::IterateAction(const po::variables_map& vm)
    : rand_eng_(std::random_device()())
    , runtime_(vm["walltime"].as<int>())
    , qdeg_(vm["qdeg"].as<int>())
    , maxfev_(vm["maxfev"].as<int>())
    , poswts_(vm["positive"].as<bool>())
    , verbose_(vm["verbose"].as<bool>())
    , lb_(vm["lb"].as<int>())
    , ntries_(3)
    , collapse_tol_(5e-2)
    , outprec_(vm["output-prec"].as<int>())
#ifdef POLYQUAD_HAVE_MPI
    , rank_(world_.rank())
    , size_(world_.size())
    , printed_npts_(vm["ub"].as<int>())
#endif
{
    // Process the tolerance
    tol_ = vm.count("tol") ? static_cast<T>(vm["tol"].as<double>())
                           : Eigen::NumTraits<T>::dummy_precision();

    update_decomps(vm["ub"].as<int>());
}

template<template<typename> class Domain, typename T>
inline void
IterateAction<Domain, T>::update_decomps(int ub)
{
    // Update the upper bound
    ub_ = ub;

    // Remove invalid decompositions from the map
    for (auto it = std::begin(drecords_); it != std::end(drecords_);)
    {
        if (it->first.first >= ub_)
            it = drecords_.erase(it);
        else
            ++it;
    }

    // Determine which point counts yield valid decompositions
    for (int i = ub_ - 1; i > lb_; --i)
    {
        const std::vector<VectorOrb> orbs = dom_.symm_decomps(i);

        for (int j = 0; j < orbs.size(); ++j)
        {
            if (drecords_.count({i, j}))
                continue;

            bool insert = true;
            for (const auto& kv : drecords_)
                if (((kv.second.orbit - orbs[j]).array() >= 0).all())
                {
                    insert = false;
                    break;
                }

            if (insert)
                drecords_[{i, j}] = {orbs[j], 0, InitResid};
        }
    }

    // Update the decomposition list
    dixs_.clear();
    for (const auto& kv : drecords_)
        dixs_.push_back(kv.first);
}

template<template<typename> class Domain, typename T>
inline void
IterateAction<Domain, T>::pick_decomp()
{
    const int sz = drecords_.size();
    const int n = std::min(5, sz);

    auto cmp = [&](const auto& i, const auto& j)
    {
        const DecompRecord& p = drecords_[i];
        const DecompRecord& q = drecords_[j];

        return p.resid*sqrt(p.ntries) < q.resid*sqrt(q.ntries);
    };

#ifdef POLYQUAD_HAVE_MPI
    if (drecords_[dixs_[0]].resid == InitResid)
        while (true)
        {
            int r = std::uniform_int_distribution(0, sz - 1)(rand_eng_);
            if (drecords_[dixs_[r]].resid == InitResid)
            {
                active_ = dixs_[r];
                return;
            }
        }
#endif

    // Sort the decompositions
    std::partial_sort(std::begin(dixs_), std::begin(dixs_) + n,
                      std::end(dixs_), cmp);

    // One third of the time go with the optimal decomposition
    if (rand_eng_() % 3 == 0)
        active_ = dixs_[0];
    // Otherwise, pick one of the top n decompositions
    else
        active_ = dixs_[std::uniform_int_distribution(0, n - 1)(rand_eng_)];
}

#ifdef POLYQUAD_HAVE_MPI
template<template<typename> class Domain, typename T>
inline void
IterateAction<Domain, T>::pump_messages()
{
    boost::optional<mpi::status> status;

    while ((status = world_.iprobe(mpi::any_source))) switch (status->tag())
    {
        case NptsTag:
        {
            int n;
            world_.recv(mpi::any_source, NptsTag, n);

            if (n < ub_)
                update_decomps(n);

            break;
        }
        case StatsTag:
        {
            std::vector<Stats> stats;
            world_.recv(mpi::any_source, StatsTag, stats);

            for (const auto& s : stats)
            {
                auto [ij, ntries, resid] = s;

                if (auto dr = drecords_.find(ij); dr != std::end(drecords_))
                {
                    dr->second.ntries += ntries;
                    dr->second.resid = std::min(dr->second.resid, resid);
                }
            }
            break;
        }
        case RuleTag:
        {
            std::pair<int, std::string> rule;
            world_.recv(mpi::any_source, RuleTag, rule);

            if (rule.first < printed_npts_)
            {
                std::cout << rule.second << std::flush;
                printed_npts_ = rule.first;
            }

            break;
        }
    }
}

template<template<typename> class Domain, typename T>
template<typename Object>
inline void
IterateAction<Domain, T>::post_message(int tag, const Object& obj)
{
    // See if any existing requests have finished
    reqs_.remove_if([](auto& rr)
    {
        return std::all_of(std::begin(rr.second), std::end(rr.second),
                           [](auto& r) {return r.test() ? true : false; });
    });

    // Create a new block of requests
    auto& [aobj, mreqs] = reqs_.emplace_back();
    aobj = obj;
    mreqs.reserve(size_ - 1);

    for (int i = 0; i < size_; ++i)
        if (i != rank_)
            mreqs.push_back(world_.isend(i, tag, std::any_cast<Object>(aobj)));
}
#endif

template<template<typename> class Domain, typename T>
inline bool
IterateAction<Domain, T>::attempt_to_minimise(int maxfev, double* rresid)
{
    // Attempt to minimise
    auto [resid, args] = dom_.minimise(maxfev);

    // Save the residual
    if (rresid)
        *rresid = static_cast<double>(resid);

    // See if we've found a rule
    if (resid < tol_)
    {
        int npts = dom_.npts();

        // See if it is superior in terms of point count
        if (npts < ub_)
        {
            // Convert the rule to a string
            auto rstr = rule_to_str(qdeg_, npts, dom_.orbits(), args, outprec_);

#ifdef POLYQUAD_HAVE_MPI
            // If we are the root rank then print our rule
            if (rank_ == 0)
            {
                std::cout << rstr << std::flush;
                printed_npts_ = npts;
            }
            // Else send it to the root rank for printing
            else
                world_.send(0, RuleTag, make_pair(npts, rstr));

            post_message(NptsTag, npts);
#else
            std::cout << rstr << std::flush;
#endif

            // Update the decompositions map
            update_decomps(npts);
        }

        // See if the rule can be further reduced
        attempt_to_reduce(args);

        return true;
    }
    else if (resid < 1e-4)
    {
        int ub = ub_;
        attempt_to_reduce(args);

        return ub_ < ub;
    }
    else
        return false;
}

template<template<typename> class Domain, typename T>
inline void
IterateAction<Domain, T>::attempt_to_reduce(const VectorXT& args)
{
    for (const auto& r : dom_.possible_reductions(args))
    {
        auto cmp = [&](const auto& s)
        {
            return (s.first - r.first).cwiseAbs().maxCoeff() == 0
                && (s.second - r.second).norm() < collapse_tol_;
        };

        // Ensure we have not considered this reduction before
        if (std::any_of(std::begin(seen_red_), std::end(seen_red_), cmp))
            continue;

        // Reconfigure the domain and verify the point count
        dom_.configure(qdeg_, poswts_, r.first);
        if (dom_.npts() <= lb_)
            continue;

        // Mark the reduction as seen
        seen_red_.push_back(r);

        dom_.seed(r.second);
        attempt_to_minimise(maxfev_);

        // If we've seen too many reductions then return
        if (seen_red_.size() > 2500)
            return;
    }
}

template<template<typename> class Domain, typename T>
inline void
IterateAction<Domain, T>::run()
{
#ifdef POLYQUAD_HAVE_MPI
    Timer tstats;
    std::vector<Stats> stats;
#endif

    while (twall_.elapsed() < runtime_)
    {
#ifdef POLYQUAD_HAVE_MPI
        pump_messages();
#endif

        // Return if no points remain
        if (!drecords_.size())
            return;

        // Pick a decomposition
        pick_decomp();

        const auto [aorbit, antries, aresid] = drecords_[active_];

        // Configure the domain for this decomposition
        dom_.configure(qdeg_, poswts_, aorbit);

        // Try to minimise this decomposition
        int ntries = ntries_;
        for (int j = 0; j < ntries; )
        {
            double resid;
            dom_.seed();
            seen_red_.clear();

            if (attempt_to_minimise(maxfev_, &resid))
                break;

            auto& r = drecords_[active_];
            if (resid < r.resid)
            {
                if (1.5*resid < r.resid && r.resid < InitResid)
                {
                    std::cout << "Rank: "<< rank_ << " ub: " << ub_ << " "
                              <<  active_.first << " " << active_.second << " "
                              << r.resid << " " << resid << std::endl;

                    ntries += 3;
                }

                r.resid = resid;
            }
            else
            {
                ++r.ntries;
                ++j;
            }

#ifdef POLYQUAD_HAVE_MPI
            pump_messages();

            if (active_.first >= ub_)
                break;
#endif
        }

#ifdef POLYQUAD_HAVE_MPI
        // Record the statistics for this decomposition
        if (auto dr = drecords_.find(active_); dr != std::end(drecords_))
            stats.emplace_back(active_, ntries, dr->second.resid);

        // Periodically, inform the other ranks of our stats
        if (tstats.elapsed() > 30)
        {
            post_message(StatsTag, stats);
            stats.clear();
            tstats.reset();
        }
#endif
    }
}

template<template<typename> class Domain, typename T>
void
process_iterate(const boost::program_options::variables_map& vm)
{
#ifdef POLYQUAD_HAVE_MPI
    mpi::environment env;
#endif

    IterateAction<Domain, T>(vm).run();
}

}

#endif /* POLYQUAD_ACTIONS_ITERATE_HPP */
