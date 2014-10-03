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
problem_setup

Functions for setting up Laplace inversion problem.

References:
Provencher Comp. Phys. Comm, 1982

"Matrices" and "vectors" refer to 2 and 1-dimensional ndarrays.
'''

import numpy as np
import numpy.linalg as linalg
from numpy import log, exp

def setup_coefficient_matrix(grid, tbase, kernel_func, kernel_kwargs, 
                             quadrature_weights, dust_term = True):
    '''
    
    '''
    n_grid = len(grid)
    n_ts = len(tbase)
    
    # kernel matrix
    Fk = np.zeros((n_ts, n_grid))
    for i in np.arange(n_ts):
        Fk[i] = kernel_func(grid, tbase[i], **kernel_kwargs)

    if dust_term:
        A = np.zeros((n_ts, n_grid + 1))
        A[:, :-1] = np.dot(Fk, np.diag(quadrature_weights))
        A[:, -1] = np.ones(n_ts)
    else:
        A = np.dot(Fk, np.diag(quadrature_weights))

    return A


def setup_nonneg(n_x):
    return np.identity(n_x), np.zeros(n_x)


def setup_regularizer(grid, n_x, skip_rows = 1):
    '''
    Define regularizing matrix R such that 
    ||Rx||^2 = \int_a^b s''(\lambda)^2 d\lambda.

    Skip skip_rows rows at beginning and end to avoid biasing solution
    by assuming 0 outside range; recommend skip_rows = 1.
    '''
    # Use generalized extended trapezoidal rule to compute integral:
    # first row: (lambda_1 - lambda_0) * s''(lambda_0)
    # next row: (lambda_2 - lambda_0) * s''(lambda_1)
    # 3rd row: (lambda_3 - lambda_1) * s''(lambda_2)
    # etc...
    # last row: (lambda_max - lambda_(max-1) * s''(lambda_max)
    # There's an overall factor of 1/2 from trapezoidal rule
    # cancelling a factor of 2 in the finite difference 2nd derivative.

    # Use generalized finite differences to estimate 2nd derivative.
    # s''(lambda_1) = ((lambda_2 - lambda_1) s(0) 
    #                   - (lambda_2 - lambda_0) s(1)
    #                   - (lambda_1 - lambda_0) s(2)) / 
    #      ((lambda_1 - lambda_0) (lambda_2 - lambda_1) (lambda_2 - lambda_0)))

    n_grid = len(grid)
    regularizer = np.zeros((n_grid, n_x))
    # first row: assume s''(lambda_-1) = 0 
    # also assume lambda_0 - lambda_-1 = lambda_1 - lambda_0
    regularizer[0,:2] = np.array([-1. / (grid[1] - grid[0]),
                                   1. / (2. * (grid[1] - grid[0]))])
    # equivalent assumptions for last row
    regularizer[-1, -2:] = np.array([1. / (grid[-1] - grid[-2]),
                                     -2. / (grid[-1] - grid[-2])])

    # fill in remainder
    for i in np.arange(1, n_grid - 1):
        regularizer[i, 
                    (i - 1):(i + 2)] = np.array([1. / (grid[i] - grid[i-1]),
                                                 -(grid[i+1] - 
                                                   grid[i-1]) / ((grid[i] - 
                                                                  grid[i-1]) * 
                                                                 (grid[i+1] - 
                                                                  grid[i])),
                                                 1. / (grid[i+1] - grid[i])])
    if skip_rows == 0:
        return regularizer
    else:
        regularizer[:skip_rows] = np.zeros((skip_rows, n_x))
        regularizer[-skip_rows:] = np.zeros((skip_rows, n_x))
        return regularizer


def dumb_regularizer(grid, n_x, skip_rows = 2):
    '''
    Testing purposes. My better regularizer causing problems?
    See Eqn. 3.12 of 1982 paper.
    '''
    n_grid = len(grid)
    n_unreg = n_x - n_grid
    regularizer = np.zeros((n_grid + 2, n_x))

    # first 2 rows
    regularizer[0, 0] = 1.
    regularizer[1, :2] = np.array([-2., 1.])
    # last 2 rows
    regularizer[-2, -(2 + n_unreg):-n_unreg] = np.array([1., -2.])
    regularizer[-1, -(1 + n_unreg)] = 1.

    # the rest
    for i in np.arange(2, n_grid):
        regularizer[i, (i-2):(i+1)] = np.array([1., -2., 1.])

    # calculate regularizer column norms for scaling
    R_col_norms = np.array([linalg.norm(regularizer[:,i], ord = 1)
                            for i in np.arange(n_x)])

    if skip_rows == 0:
        return regularizer #, R_col_norms
    else:
        regularizer[:skip_rows] = np.zeros((skip_rows, n_x))
        regularizer[-skip_rows:] = np.zeros((skip_rows, n_x))
        return regularizer #, R_col_norms


def setup_grid(grid_min, grid_max, n_grid, type = 'log'):
    '''
    Set up grid of points over which solution is computed.
    '''
    if type == 'log':
        grid = np.logspace(log(grid_min), log(grid_max), n_grid, 
                           base = exp(1))
        dh = (np.log(grid_max) - np.log(grid_min)) / (n_grid - 1)
        dhdx = 1. / grid
    elif type == 'linear':
        grid = np.linspace(grid_min, grid_max, n_grid)
        dh = grid[1] - grid[0]
        dhdx = np.ones(n_grid)
    else:
        raise NotImplementedError

    return grid, dh, dhdx


def setup_quadrature(grid_x, dh, dhdx, type = 'simpson'):
    n_pts = len(grid_x)
    if type == 'simpson':
        if n_pts % 2 == 0: # even number of points
            # n-1 pts like odd case
            weights = np.ones(n_pts - 1) * 2/3. + \
                np.arange(n_pts - 1) % 2 * 2./3.
            weights[0] = 1./3.
            weights[-1] = 5./6.
            # do last point by trapezoidal
            weights = np.append(weights, 0.5)
        else: # odd, regular "extended Simpson's rule"
            # [1/3, 4/3, 2/3, 4/3, 2/3, ..., 4/3, 1/3]
            weights = np.ones(n_pts) * 2./3. + np.arange(n_pts) % 2 * 2./3.
            weights[0] = 1./3.
            weights[-1] = 1./3.
    elif type == 'trapezoidal':
        weights = np.ones(n_pts)
        weights[0] = 0.5
        weights[-1] = 0.5
    else:
        raise NotImplementedError
    
    return dh * weights / dhdx # approximate dx ~ dh / (dh/dx)


def setup_weights(y):
    '''
    See Eq. 2 on p. 4.1.2.8-1 of CONTIN manual. 
    Return 1/sigma_k = sqrt(W_k), which multiplies each element of y_k
    and each row of coefficient matrix A_kj.
    '''
    return 2. * y / np.sqrt(y**2 + 1)


def nonneg_sqrt(arr):
    nnsqrt = np.sqrt(np.abs(arr))
    nnsqrt[arr<0] *= -1
    return nnsqrt
