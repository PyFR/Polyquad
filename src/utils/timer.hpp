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

#ifndef POLYQUAD_UTILS_TIMER_HPP
#define POLYQUAD_UTILS_TIMER_HPP

#include <chrono>

namespace polyquad {

class Timer
{
public:
    Timer() : start_(now())
    {}

    double elapsed() const
    { return now() - start_; }

    void reset()
    { start_ = now(); }

private:
    static double now();

    double start_;
};

inline double
Timer::now()
{
    using namespace std::chrono;

    auto t = high_resolution_clock::now().time_since_epoch();
    auto d = duration_cast<duration<double>>(t);

    return d.count();
}

}

#endif /* POLYQUAD_UTILS_TIMER_HPP */
