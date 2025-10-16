##################################################
# Main build file for Polyquad
##################################################
cmake_minimum_required(VERSION 3.15)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(polyquad VERSION 1.1.0 DESCRIPTION "Polyquad" LANGUAGES CXX)

include(CheckIncludeFiles)
include(GNUInstallDirs)

# Add our custom CMake module dir
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

##################################################
# Default build settings
##################################################
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# C++17 and warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(UNIX)
    add_compile_options(-Wno-deprecated)
endif()

##################################################
# Source files
##################################################
SET(POLYQUAD_HEADERS
    src/actions/eval.hpp
    src/actions/expand.hpp
    src/actions/find.hpp
    src/actions/iterate.hpp
    src/shapes/base.hpp
    src/shapes/hex.hpp
    src/shapes/quad.hpp
    src/shapes/tet.hpp
    src/shapes/tri.hpp
    src/shapes/pyr.hpp
    src/shapes/pri.hpp
    src/utils/io.hpp
    src/utils/eigen_addons.hpp
    src/utils/ortho_poly.hpp
    src/utils/serialization.hpp
    src/utils/timer.hpp)

##################################################
# Executable
##################################################
add_executable(polyquad src/main.cpp ${POLYQUAD_HEADERS})

##################################################
# Internal includes, these must come before the
# located dependencies!
##################################################
target_include_directories(polyquad
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)

##################################################
# External dependencies
##################################################
find_package(Eigen3 3.4 REQUIRED CONFIG)
target_link_libraries(polyquad PRIVATE Eigen3::Eigen)
get_target_property(EIGEN3_INCLUDE Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(polyquad PRIVATE ${EIGEN3_INCLUDE}/unsupported)

set(POLYQUAD_BOOST_LIBS program_options serialization)
find_package(Boost 1.74 REQUIRED CONFIG COMPONENTS ${POLYQUAD_BOOST_LIBS})
target_link_libraries(polyquad PRIVATE Boost::program_options Boost::serialization)

##################################################
# MPI
##################################################
find_package(MPI)
if(TARGET MPI::MPI_CXX)
    find_package(Boost 1.74 CONFIG COMPONENTS mpi)
    if(TARGET Boost::mpi)
        target_link_libraries(polyquad PRIVATE Boost::mpi MPI::MPI_CXX)
        set(POLYQUAD_HAVE_MPI 1)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(POLYQUAD_USE_MULTIPRECISION "Enable multi-precision arithmetic support" OFF)
else()
    option(POLYQUAD_USE_MULTIPRECISION "Enable multi-precision arithmetic support" ON)
endif()

if(POLYQUAD_USE_MULTIPRECISION)
    set(POLYQUAD_HAVE_MULTIPRECISION 1)
endif()

##################################################
# Build type and optimisation
##################################################
if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(polyquad PRIVATE -O3 -march=native)
endif()

##################################################
# Date & compiler info
##################################################
include(Today)
today(POLYQUAD_BUILD_DATE)

#INCLUDE(GetGitRevisionDescription)
#GET_GIT_HEAD_REVISION(GIT_REFSPEC POLYQUAD_GIT_SHA1)

##################################################
# Config.h
##################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)
