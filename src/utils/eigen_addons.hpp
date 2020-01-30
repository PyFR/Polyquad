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

EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Matrix(const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d, const Scalar& e)
{
    Base::_check_template_params();
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 5)
    m_storage.data()[0] = a;
    m_storage.data()[1] = b;
    m_storage.data()[2] = c;
    m_storage.data()[3] = d;
    m_storage.data()[4] = e;
}

EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Matrix(const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d, const Scalar& e, const Scalar& f)
{
    Base::_check_template_params();
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 6)
    m_storage.data()[0] = a;
    m_storage.data()[1] = b;
    m_storage.data()[2] = c;
    m_storage.data()[3] = d;
    m_storage.data()[4] = e;
    m_storage.data()[5] = f;
}
