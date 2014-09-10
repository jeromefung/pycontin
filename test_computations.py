# Copyright 2014, Jerome Fung 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Test contin math.
'''

import numpy as np
from numpy.testing import assert_allclose
from computations import ldp_lawson_hanson, reduce_A_qr

def test_ldp():
    # see eqn. 23.54 on p. 171
    Gtilde = np.array([[-0.207, 2.558],
                       [-0.392, -1.351], 
                       [0.599, -1.206]])
    htilde = np.array([-1.3, -0.084, 0.384])
    x, binding, success = ldp_lawson_hanson(Gtilde, htilde)
    # see solution on p. 192 of LH
    assert_allclose(x, np.array([0.127, -0.255]), rtol = 2e-3)

def test_eqn_A_1():
    # simple Householder example
    # http://www.math.sjsu.edu/~foster/m143m/least_squares_via_Householder.pdf
    A = np.array([[3., -2.],
                  [0., 3.],
                  [4., 4.]])
    b = np.array([3., 5., 4.])
    
    C, eta = reduce_A_qr(A, b)
    assert_allclose(C, np.array([[-5., -2.], 
                                 [0., -5.]]))
    assert_allclose(eta, np.array([-5., -3.]))


