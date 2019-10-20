##################################################
# Main build file for Polyquad
##################################################
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(CheckIncludeFiles)

PROJECT(polyquad)

# Add our custom CMake module dir
SET(CMAKE_MODULE_PATH ${polyquad_SOURCE_DIR}/cmake)

SET(POLYQUAD_VERSION_MAJOR 0)
SET(POLYQUAD_VERSION_MINOR 1)
SET(POLYQUAD_VERSION "${POLYQUAD_VERSION_MAJOR}.${POLYQUAD_VERSION_MINOR}")

##################################################
# Source files
##################################################
SET(POLYQUAD_HEADERS
    src/actions/eval.hpp
    src/actions/expand.hpp
    src/actions/find.hpp
    src/shapes/base.hpp
    src/shapes/hex.hpp
    src/shapes/quad.hpp
    src/shapes/tet.hpp
    src/shapes/tri.hpp
    src/shapes/pyr.hpp
    src/shapes/pri.hpp
    src/utils/io.hpp
    src/utils/jacobi_poly.hpp
    src/utils/serialization.hpp
    src/utils/timer.hpp)

##################################################
# Executable
##################################################
ADD_EXECUTABLE(polyquad src/main.cpp ${POLYQUAD_HEADERS})

##################################################
# Internal includes, these must come before the
# located dependencies!
##################################################
INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

##################################################
# External dependencies
##################################################
SET(POLYQUAD_BOOST_LIBS
    program_options
    serialization)

# Find Boost
FIND_PACKAGE(Boost REQUIRED ${POLYQUAD_BOOST_LIBS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(polyquad ${Boost_LIBRARIES})

# Find Eigen 3
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported)

# Find MPFR and the mpreal wrapper
FIND_PACKAGE(MPFR)
IF(MPFR_FOUND)
    FIND_FILE(POLYQUAD_HAVE_MPREAL mpreal.h)
    IF (POLYQUAD_HAVE_MPREAL)
       INCLUDE_DIRECTORIES(${MPFR_INCLUDE_DIR})
       LINK_DIRECTORIES(${MPFR_LIBRARY_DIR})
       TARGET_LINK_LIBRARIES(polyquad ${MPFR_LIBRARIES})
    ENDIF()
ENDIF()

##################################################
# MPI
##################################################
FIND_PACKAGE(MPI)
IF(MPI_CXX_FOUND)
    FIND_PACKAGE(Boost COMPONENTS mpi)
    IF(Boost_MPI_FOUND)
        SET(CMAKE_CXX_COMPILE_FLAGS
            ${CMAKE_CXX_COMPILE_FLAGS}
            ${MPI_COMPILE_FLAGS})
        SET(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
        INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
        TARGET_LINK_LIBRARIES(polyquad
                              ${Boost_MPI_LIBRARY}
                              ${MPI_CXX_LIBRARIES})
        SET(POLYQUAD_HAVE_MPI 1)
    ENDIF()
ENDIF()

# Avoid ';' in the link flags
STRING(REPLACE ";" " " CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS}")

##################################################
# C++14
##################################################
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-deprecated")
ENDIF()

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-fabi-version=0 HAVE_ABI_VERSION)
IF(HAVE_ABI_VERSION)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=0")
ENDIF()

##################################################
# Build type
##################################################
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

##################################################
# Optimisation
##################################################

# Auto-optimise for CPU arch for non-debug builds
IF(NOT CMAKE_BUILD_TYPE MATCHES Debug)
  ADD_DEFINITIONS("-Ofast -march=native")
ENDIF()

# For release builds remove all debugging info
IF(CMAKE_BUILD_TYPE MATCHES Release)
  ADD_DEFINITIONS("-DNDEBUG")
ENDIF()

##################################################
# Date & compiler info
##################################################
INCLUDE(Today)
TODAY(POLYQUAD_BUILD_DATE)

#INCLUDE(GetGitRevisionDescription)
#GET_GIT_HEAD_REVISION(GIT_REFSPEC POLYQUAD_GIT_SHA1)

##################################################
# Config.h
##################################################
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)
