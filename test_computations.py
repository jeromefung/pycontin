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
from numpy.linalg import norm
from numpy.testing import assert_allclose
from computations import ldp_lawson_hanson, reduce_A_qr, solve_fixed_alpha
from problem_setup import setup_grid, setup_quadrature, setup_regularizer
import scipy.optimize

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

def test_quadrature():
    # 4 cases: linear spacing, log spacing, simpson, trapezoidal
    grid_min = 1.
    grid_max = 2.
    n_pts = 101
    
    linear_grid, lin_dh, lin_dhdx = setup_grid(grid_min, grid_max, n_pts, 
                                               type = 'linear')
    quad_coeffs_linear_trap = setup_quadrature(linear_grid, lin_dh, lin_dhdx,
                                               type = 'trapezoidal')
    quad_coeffs_linear_simp = setup_quadrature(linear_grid, lin_dh, lin_dhdx,
                                               type = 'simpson')
    
    log_grid, log_dh, log_dhdx = setup_grid(grid_min, grid_max, n_pts,
                                            type = 'log')
    quad_coeffs_log_trap = setup_quadrature(log_grid, log_dh, log_dhdx,
                                            type = 'trapezoidal')
    quad_coeffs_log_simp = setup_quadrature(log_grid,log_dh, log_dhdx,
                                            type = 'simpson')

    # tolerance where it is b/c of inherent approximations in 
    # setting up quadrature grid w/non-uniform spacing, becomes exact as 
    # n_pts -> infinity.
    assert_allclose(np.array([(quad_coeffs_linear_trap * linear_grid).sum(),
                              (quad_coeffs_log_trap * log_grid).sum(),
                              (quad_coeffs_linear_simp * linear_grid**2).sum(),
                              (quad_coeffs_log_simp * log_grid**2).sum()]),
                    np.array([1.5, 1.5, 7./3., 7./3.]), rtol = 2e-5)


def test_regularizer():
    '''
    Check the integral of the second derivative of a function.
    '''
    grid_min = 1.
    grid_max = 1.1
    n_pts = 100

    grid, dh, dhdx = setup_grid(grid_min, grid_max, n_pts, type = 'log')
    s = grid**3

    regularizer = setup_regularizer(grid, n_pts) 
    Rs = np.dot(regularizer, s)
    regularized_integral = Rs.sum()
    gold = (3. * (grid[-1]**2 - grid[1]**2))

    assert_allclose(regularized_integral, gold, rtol = 1e-3)
    
 
def test_lh_lsi_example():
    '''
    Test regularized minimization for alpha = 0.
    minimize |Ex - f| subject to Gx >= h.
    See Problem LSI example on p. 169 - 173 of Lawson/Hanson (sec. 23.7) 
    '''
    E = np.array([[0.25, 1.],
                  [0.5, 1.],
                  [0.5, 1.],
                  [0.8, 1.]])
    f = np.array([0.5, 0.6, 0.7, 1.2])
    G = np.array([[1., 0.],
                  [0., 1.],
                  [-1., -1.]])
    h = np.array([0., 0., -1.])

    identity = np.diag(np.ones(2))
    x, err, infodict, int_res = solve_fixed_alpha(E, f, 1e-10,
                                                  identity, 
                                                  G, h, True)
    #print x
    #print infodict
    #print int_res

    x2, err2, infodict2 = solve_fixed_alpha(E, f, 1e-10, 
                                            np.ones(4).reshape((2,2,)),
                                            G, h, False)
    gold_x = np.array([0.621, 0.379]) # see L/H p. 172
    assert_allclose(x, gold_x, rtol = 9e-4)
    assert_allclose(x2, gold_x, rtol = 9e-4)
