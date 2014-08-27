#include "config.h"

#include "quad.hpp"
#include "tri.hpp"
#include "tet.hpp"
#include "hex.hpp"
#include "pri.hpp"
#include "pyr.hpp"
#include "util.hpp"

#include <boost/algorithm/string/join.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#ifdef POLYQUAD_HAVE_MPI
# include <boost/mpi.hpp>
#endif
#include <boost/program_options.hpp>

#include <Eigen/Dense>
#ifdef POLYQUAD_HAVE_MPREAL
# include <mpreal.h>
# include <Eigen/MPRealSupport>
#endif

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace po = boost::program_options;
using namespace polyquad;

template<typename Derived>
bool
compare_pts(const Eigen::MatrixBase<Derived>& a,
            const Eigen::MatrixBase<Derived>& b,
            const typename Eigen::MatrixBase<Derived>::Scalar& tol)
{
    const auto& br = b.rowwise();

    for (int i = 0; i < a.rows(); ++i)
        if ((br - a.row(i)).rowwise().squaredNorm().minCoeff() > tol)
            return false;

    return true;
}

template<template<typename> class Domain, typename T>
void
erase_duplicates(const Domain<T>& dom,
                 std::vector<typename Domain<T>::VectorXT>& m,
                 const T& tol)
{
    typedef typename Domain<T>::MatrixPtsT MatrixPtsT;

    for (auto it = m.begin(); it != m.end();)
    {
        bool seen = false;

        MatrixPtsT pts_a(dom.npts(), dom.ndim());
        MatrixPtsT pts_b(dom.npts(), dom.ndim());

        dom.expand(*it, pts_a);

        for (auto iit = m.begin(); iit != it && !seen; ++iit)
        {
            dom.expand(*iit, pts_b);

            seen = compare_pts(pts_a, pts_b, tol);
        }

        if (seen)
            it = m.erase(it);
        else
            ++it;
    }
}

template<typename D, typename R>
void
print_compact(std::ostream& out, int outprec, const D& decomp, const R& args)
{
    const Eigen::IOFormat fmt(outprec, Eigen::DontAlignCols);

    out << decomp.transpose().format(fmt) << "\n";
    out << args.format(fmt) << "\n";
}

template<template<typename> class Domain, typename T>
void
process_find(const po::variables_map& vm)
{
    typedef typename Domain<T>::MatrixXT MatrixXT;
    typedef typename Domain<T>::VectorXT VectorXT;
    typedef typename Domain<T>::MatrixPtsT MatrixPtsT;
    typedef typename Domain<T>::VectorOrb VectorOrb;

#ifdef POLYQUAD_HAVE_MPI
    boost::mpi::environment env;
    boost::mpi::communicator world;

    const int rank = world.rank();
#else
    const int rank = 0;
#endif

    // Pinciple domain
    Domain<T> dom;

    // Desired quadrature degree and point count
    const int qdeg = vm["qdeg"].as<int>();
    const int npts = vm["npts"].as<int>();
    const int maxfev = vm["maxfev"].as<int>();
    const double runtime = vm["walltime"].as<int>();

    // Floating point tolerance and output precision
    const T tol = vm.count("tol") ? static_cast<T>(vm["tol"].as<double>())
                                  : Eigen::NumTraits<T>::dummy_precision();
    const int outprec = vm["output-prec"].as<int>();

    // Flags
    const bool poswts = vm["positive"].as<bool>();
    const bool verbose = vm["verbose"].as<bool>();

    T norm;
    VectorXT args;

    // Decompose npts into symmetric orbital configurations
    auto orbits = dom.symm_decomps(npts);

    // Discovered rules for each configuration
    std::vector<std::vector<VectorXT>> rules(orbits.size());

    for (int i = 0; i < orbits.size(); ++i)
    {
        if (verbose && rank == 0)
            std::cerr << "Decomposition: " << (i + 1) << "/"
                      << orbits.size() << ": " << orbits[i].transpose()
                      << std::endl;

        dom.configure(qdeg, orbits[i]);

        Timer t;
        do
        {
            // Seed the orbit
            dom.seed();

            // Attempt to minimise
            std::tie(norm, args) = dom.minimise(maxfev);

            // See if the minimisation was successful
            if (norm < tol && rules[i].size() < 1000
             && (!poswts || (dom.wts(args).minCoeff() > 0)))
                rules[i].push_back(args);
        } while (orbits.size()*t.elapsed() < runtime);

        erase_duplicates(dom, rules[i], tol);

#ifdef POLYQUAD_HAVE_MPI
        if (rank == 0)
        {
            std::vector<std::vector<VectorXT>> grules;
            boost::mpi::gather(world, rules[i], grules, 0);

            rules[i].clear();
            for (const auto& rv : grules)
                std::copy(rv.begin(), rv.end(), std::back_inserter(rules[i]));

            erase_duplicates(dom, rules[i], tol);
        }
        else
        {
            boost::mpi::gather(world, rules[i], 0);
            rules[i].clear();
        }
#endif

        if (rank == 0)
        {
            if (verbose)
                std::cerr << "Rule count: " << rules[i].size() << std::endl;

            for (const auto& r : rules[i])
            {
                std::cout << "# Rule degree: " << qdeg << "\n";
                print_compact(std::cout, outprec, orbits[i], r);
                std::cout << std::endl;
            }
        }
    }
}

template<template<typename> class Domain, typename T>
void
process_eval(const po::variables_map& vm)
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
            dom.configure(qdeg, orb);

            // Now read in the arguments (i.e, the rule)
            VectorXT args = VectorXT::Zero(dom.ndof());
            ifs >> args;

            // Compute the residual norm
            VectorXT resid(dom.nbfn());
            dom.wts(args, &resid);
            T rnorm = resid.norm();

            if (verbose)
                std::cerr << resid;

            std::cout << "# Rule degree: " << qdeg << "\n"
                      << "# Residual norm: " << rnorm << "\n";

            if (refine)
            {
                // Take the current rule to be the seed for minimisation
                dom.seed(args);

                std::tie(rnorm, args) = dom.minimise(maxfev);

                if (verbose)
                    std::cerr << " (" << rnorm << ")";

                std::cout << "# Refined residual norm: " << rnorm << "\n";
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

template<template<typename> class Domain, typename T>
void
process_expand(const po::variables_map& vm)
{
    typedef typename Domain<T>::MatrixXT MatrixXT;
    typedef typename Domain<T>::VectorXT VectorXT;
    typedef typename Domain<T>::MatrixPtsT MatrixPtsT;
    typedef typename Domain<T>::VectorOrb VectorOrb;

    // Pinciple domain
    Domain<T> dom;

    // Quadrature degree of the rule and the max number of fn evals
    const int qdeg = vm["qdeg"].as<int>();

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

template<template<typename> class Domain, typename T>
void process_dispatch(const po::variables_map& vm)
{
    const std::string& action = vm["action"].as<std::string>();

    if (action == "find")
        process_find<Domain, T>(vm);
    else if (action == "eval")
        process_eval<Domain, T>(vm);
    else if (action == "expand")
        process_expand<Domain, T>(vm);
}

int main(int argc, char *argv[])
{
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
#ifdef POLYQUAD_HAVE_MPREAL
        { {"tri",  "mpreal"}, &process_dispatch<TriDomain,  mpfr::mpreal> },
        { {"quad", "mpreal"}, &process_dispatch<QuadDomain, mpfr::mpreal> },
        { {"hex",  "mpreal"}, &process_dispatch<HexDomain,  mpfr::mpreal> },
        { {"tet",  "mpreal"}, &process_dispatch<TetDomain,  mpfr::mpreal> },
        { {"pri",  "mpreal"}, &process_dispatch<PriDomain,  mpfr::mpreal> },
        { {"pyr",  "mpreal"}, &process_dispatch<PyrDomain,  mpfr::mpreal> },
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
        ("qdeg,q", po::value<int>()->required(), "Target quadrature degree")
        ("maxfev,m", po::value<int>()->default_value(-1, ""),
         "Maximum number of objective function evaluations")
#ifdef POLYQUAD_HAVE_MPREAL
        ("mpfr-bits",
         po::value<int>()->default_value(256)
                         ->notifier(mpfr::mpreal::set_default_prec),
         "Base precision for MPFR reals")
#endif
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
        ("walltime,w", po::value<int>()->default_value(300),
         "Approximate run time in seconds")
        ("positive,p", po::value<bool>()->default_value(false)->zero_tokens(),
         "Enforce positivity of weights");

    // Rule evaluation specific options
    action_opts.insert({"eval", std::string("Eval action options")});
    action_opts["eval"].add_options()
        ("refine,r", po::value<bool>()->default_value(false)->zero_tokens(),
         "Attempt to refine the rule");

    // Rule expansion specific options
    action_opts.insert({"expand", std::string("Expand action options")});
    action_opts["expand"].add_options()
        ("header,x", po::value<bool>()->default_value(false)->zero_tokens(),
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
                      << " built on " POLYQUAD_BUILD_DATE "\n";
            return 0;
        }

        // Handle --help
        if (vm.count("help") || !vm.count("action"))
        {
            std::cout << "Usage: polyquad [OPTION...] <action>\n\n"
                         "<action> is one of: find, eval, expand\n";
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
