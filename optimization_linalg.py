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
optimization_linalg

Functions for optimization and linear algebra

References:
Lawson & Hanson
Provencher Comp. Phys. Comm, 1982

"Matrices" and "vectors" refer to 2 and 1-dimensional ndarrays.
'''

import numpy as np
import numpy.linalg
from scipy.optimize import nnls

def ldp_lawson_hanson(G, h):
    '''
    Finds vector x in R^n of minimum Euclidean norm satisfying
    G x >= h, where G is a m x n real matrix and h is an element of
    R^m.
    
    Implements Lawson & Hanson Algorithm LDP (23.27), p. 165.

    Parameters
    ----------
    G:
        Matrix (m x n) on LHS of inequality constraints
    h:
        Vector (m) on RHS of inequality constraints

    Returns
    -------
    x:
        Vector (n) of optimal solution, if found
    binding_constraints:
        ndarray of indices of elements of x where the inequality 
        constraint is binding
    success:
        Boolean (True) if solution found
    '''
    # dimensions
    m, n = G.shape

    # construct augmented matrix E
    E = np.vstack((G.transpose(), h))
    # construct vector f
    f = np.concatenate((np.zeros(n), np.ones(1)))
    # solve non-negative least squares problem min || E u - f ||
    u, u_residuals = nnls(E, f)

    # compute r
    r = np.dot(E, u) - f

    if u_residuals == 0:
        # constraints are incompatible
        success = False
        return success
    else:
        success = True
        # see discussion of set S on p. 166 and eqn. 23.30 (Kuhn-Tucker)
        binding_constraints = np.where(u[:-1] > 0)[0]
        # compute LDP solution vector
        x = -r[:-1] / r[-1] 
        return x, binding_constraints, success


def reduce_A_qr(A, y):
    '''
    Perform a QR decomposition via Householder transformations
    to reduce coefficient matrix M_e A (N_y x N_x) and data vector 
    Me y to N_x x N_x matrix C and N_x dimensional eta.
   
    See Eqn. A.1 of Provencher 1981.
    '''
    orthog, C = numpy.linalg.qr(A)
    eta = np.dot(orthog.transpose(), y)
    return C, eta



def sv_decompose_regularizer(R):
    '''
    Compute matrices U, H1, and Z in eqn. A.7

    Parameters
    ----------
    R:
        Regularizer matrix, ndarray (n_reg, n_x). If there have been 
        equality constraints, assume they have been eliminated such that
        the input matrix is RK2 (n_reg x n_xe).
    
    Returns
    -------
    H1_inv:
        ndarray (n_x, n_x), diagonal
    Z:
        ndarray (n_x, n_x)
    '''
    n_reg, n_x = R.shape

    # n_reg must be >= n_x
    # add rows of zeros to make n_reg = n_x if needed
    if n_reg < n_x:
        R = np.vstack((R, np.zeros((n_x - n_reg, n_x))))
        n_reg = n_x

    # see svd docstring. H1 is 1d array of singular values
    U, H1, Ztr = np.linalg.svd(R)

    # check if singular values are too close to epsilon
    # increase to fraction of largest SV
    machine_epsilon = np.finfo(np.float64).eps
    abs_singular_vals = np.abs(H1)
    smallest_sv = np.sqrt(machine_epsilon) * abs_singular_vals.max()
    sign_arr = np.sign(H1) # sign(0) = 0
    sign_arr[sign_arr==0] = 1.
    H1 = H1 + (abs_singular_vals < 
               smallest_sv) * (smallest_sv - abs_singular_vals) * sign_arr

    #import pdb; pdb.set_trace()
    return np.diag(1./H1), Ztr.transpose()


def sv_decompose_coeffs(CK2ZH1inv):
    '''
    eqn a.15
    '''
    Q, S, Wtr = np.linalg.svd(CK2ZH1inv)
    return Q, S, Wtr.transpose()

