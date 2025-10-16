Polyquad
========

About
-----

Polyquad is an open-source C++ application for finding symmetric
quadrature rules suitable for use with the finite element method.
Both the target strength of the quadrature and the number of points
are configurable. Polyquad is parallelised using MPI and can refine
rules to an arbitrary degree of numerical precision.

Authors
-------

See the AUTHORS file.

License
-------

Polyquad is release under the GNU GPL v3.0 license.  Please see
COPYING for further details.

Dependencies
------------

In order to build polyquad it is necessary to have first installed:

- [CMake](https://www.cmake.org/) 3.15 or later
- [Eigen](https://eigen.tuxfamily.org/) 3.4 or later
- [Boost](https://www.boost.org/) 1.74 or later, specifically
    - mpi (optional)
    - multiprecision (optional)
    - program_options
    - serialization

If Boost MPI and a suitable compiler are found then polyquad will be
able to run on clusters.  As polyquad makes use of advanced template
metaprogramming features it is important to build it using a compiler
that supports the C++17 standard.  As of the time of writing the
following compilers are known to have successfully built polyquad:

- GCC 9.2
- Clang 8

Building polyquad can require in excess of four gigabytes of main
memory and several minutes of CPU time.

Building polyquad
-----------------

After cloning the repository polyquad can be built by issuing the
following commands:

    $ mkdir build
    $ cd build
    $ cmake -G"Unix Makefiles" ../
    $ make

for further information please consult the CMake documentation and the
CMakeLists.txt file.

Running polyquad
----------------

Usage instructions can be obtained by running:

    $ ./polyquad --help

a simple example we consider using polyquad to search for strength 5
rules inside of a tetrahedron using 15 points:

    $ ./polyquad find -s tet -q5 -n15 -V -p > rules.txt

where -V enables verbose output and -p requires all rules to have
positive weights.  We can then convert the rules from the internal
representation used by polyquad to a standard tabular format by
issuing:

    $ ./polyquad expand -s tet -q5 < rules.txt > rules-tab.txt
