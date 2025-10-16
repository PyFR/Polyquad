/*
    This file is part of polyquad.
    Copyright (C) 2025  Freddie Witherden <freddie@witherden.org>

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

#include "actions/eval.hpp"
#include "actions/expand.hpp"
#include "actions/find.hpp"
#include "actions/iterate.hpp"
#include "shapes/quad.hpp"
#include "shapes/tri.hpp"
#include "shapes/tet.hpp"
#include "shapes/hex.hpp"
#include "shapes/pri.hpp"
#include "shapes/pyr.hpp"
#include "utils/io.hpp"

#include <boost/algorithm/string/join.hpp>
#ifdef POLYQUAD_HAVE_MULTIPRECISION
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/eigen.hpp>
#endif
#include <boost/program_options.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace po = boost::program_options;
using namespace polyquad;

template<template<typename> class Domain, typename T>
void process_dispatch(const po::variables_map& vm)
{
    const std::string& action = vm["action"].as<std::string>();

    if (action == "find")
        process_find<Domain, T>(vm);
    else if (action == "iterate")
        process_iterate<Domain, T>(vm);
    else if (action == "eval")
        process_eval<Domain, T>(vm);
    else if (action == "expand")
        process_expand<Domain, T>(vm);
}

int main(int argc, const char *argv[])
{
#ifdef POLYQUAD_HAVE_MULTIPRECISION
    typedef boost::multiprecision::cpp_bin_float_100 bfloat;
#endif
    typedef std::pair<std::string, std::string> shape_key;
    typedef void (*process_fn)(const po::variables_map&);

    std::map<shape_key, process_fn> shape_disp =
    {
        { {"tri",  "double"}, &process_dispatch<TriDomain,  double> },
        { {"quad", "double"}, &process_dispatch<QuadDomain, double> },
        { {"hex",  "double"}, &process_dispatch<HexDomain,  double> },
        { {"tet",  "double"}, &process_dispatch<TetDomain,  double> },
        { {"pri",  "double"}, &process_dispatch<PriDomain,  double> },
        { {"pyr",  "double"}, &process_dispatch<PyrDomain,  double> },
#ifdef POLYQUAD_HAVE_MULTIPRECISION
        { {"tri",  "bfloat"}, &process_dispatch<TriDomain,  bfloat> },
        { {"quad", "bfloat"}, &process_dispatch<QuadDomain, bfloat> },
        { {"hex",  "bfloat"}, &process_dispatch<HexDomain,  bfloat> },
        { {"tet",  "bfloat"}, &process_dispatch<TetDomain,  bfloat> },
        { {"pri",  "bfloat"}, &process_dispatch<PriDomain,  bfloat> },
        { {"pyr",  "bfloat"}, &process_dispatch<PyrDomain,  bfloat> },
#endif
    };

    std::set<std::string> shapes, dtypes;
    for (const auto& it : shape_disp)
    {
        shapes.insert(it.first.first);
        dtypes.insert(it.first.second);
    }

    // Generic options
    po::options_description generic_opt("Common options");
    generic_opt.add_options()
        ("help,h", "Displays this information")
        ("version,v", "Displays version information")
        ("verbose,V", po::value<bool>()->default_value(false)->zero_tokens(),
         "Enable verbose output")
        ("shape,s", po::value<std::string>()->required(),
         ("Shape: " + boost::algorithm::join(shapes, ", ")).c_str())
        ("dtype,d", po::value<std::string>()->default_value("double"),
         ("Data type: " + boost::algorithm::join(dtypes, ", ")).c_str())
        ("walltime,w", po::value<int>()->default_value(300),
         "Approximate run time in seconds")
        ("qdeg,q", po::value<int>()->required(), "Target quadrature degree")
        ("maxfev,m", po::value<int>()->default_value(1000000, ""),
         "Maximum number of objective function evaluations")
        ("positive,p", po::value<bool>()->default_value(false)->zero_tokens(),
         "Enforce positivity of weights")
        ("output-prec,P",
         po::value<int>()->default_value(Eigen::FullPrecision, ""),
         "Output precision")
        ("tol,t", po::value<double>(), "Tolerance");

    // Hidden (positional) options
    po::options_description hidden_opt("Hidden options");
    hidden_opt.add_options()
        ("action", po::value<std::string>(), "Action");

    po::positional_options_description pos_opt;
    pos_opt.add("action", 1);

    std::map<std::string, po::options_description> action_opts;

    // Rule finding specific options
    action_opts.insert({"find", std::string("Find action options")});
    action_opts["find"].add_options()
        ("npts,n", po::value<int>()->required(), "Desired number of points")
        ("nprelim,f", po::value<int>()->default_value(8),
         "For two phase runs how many preliminary runs to perform")
        ("two-phase,b", po::value<bool>()->default_value(false)->zero_tokens(),
         "Employ a two phase approach to rule finding");

    // Rule iteration specific options
    action_opts.insert({"iterate", std::string("Iterate action options")});
    action_opts["iterate"].add_options()
        ("limits,L", po::value<std::string>(), "Orbit limits")
        ("lb,l", po::value<int>()->required(), "Lower bound")
        ("ub,u", po::value<int>()->required(), "Upper bound");

    // Rule evaluation specific options
    action_opts.insert({"eval", std::string("Eval action options")});
    action_opts["eval"].add_options()
        ("refine,r", po::value<bool>()->default_value(false)->zero_tokens(),
         "Attempt to refine the rule");

    // Rule expansion specific options
    action_opts.insert({"expand", std::string("Expand action options")});
    action_opts["expand"].add_options()
        ("header,x", po::value<bool>()->default_value(true)->zero_tokens(),
         "Output a header with each rule");

    // Collate
    po::options_description all_opt;
    all_opt.add(generic_opt).add(hidden_opt);
    for (const auto& aopt : action_opts)
        all_opt.add(aopt.second);

    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .options(all_opt).positional(pos_opt).run(),
                  vm);

        // Handle --version
        if (vm.count("version"))
        {
            std::cout << "polyquad " POLYQUAD_VERSION
                         " built on " POLYQUAD_BUILD_DATE "\n";
#ifdef POLYQUAD_HAVE_MPI
            std::cout << "MPI support: ENABLED\n";
#else
            std::cout << "MPI support: DISABLED\n";
#endif
            return 0;
        }

        // Handle --help
        if (vm.count("help") || !vm.count("action"))
        {
            std::cout << "Usage: polyquad [OPTION...] <action>\n\n"
                         "<action> is one of: find, iterate, eval, expand\n";
            std::cout << generic_opt << "\n";

            for (const auto& aopt : action_opts)
                std::cout << aopt.second  << "\n";
            return 0;
        }

        // Obtain the action
        const std::string action = vm["action"].as<std::string>();
        vm.clear();

        // Validate the action and its arguments
        if (action_opts.count(action))
        {
            po::options_description target_opt;
            target_opt.add(generic_opt)
                      .add(hidden_opt)
                      .add(action_opts[action]);

            po::store(po::command_line_parser(argc, argv)
                      .options(target_opt).positional(pos_opt).run(),
                      vm);
            po::notify(vm);

            // Shape
            const std::string& shape = vm["shape"].as<std::string>();
            if (!shapes.count(shape))
            {
                throw std::invalid_argument("Invalid shape");
            }

            // Data type
            const std::string& dtype = vm["dtype"].as<std::string>();
            if (!dtypes.count(dtype))
            {
                throw std::invalid_argument("Invalid data type");
            }

            // Dispatch
            shape_disp[{shape, dtype}](vm);
        }
        else
        {
            throw std::invalid_argument("Bad action");
        }
    }
    catch (std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
