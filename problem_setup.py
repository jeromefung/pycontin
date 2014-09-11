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

def setup_regularizor():
    '''
    '''
    pass


def setup_grid(grid_min, grid_max, n_grid, type = 'log'):
    '''
    '''
    if type == 'log':
        grid = np.logspace(grid_min, grid_max, n_grid)
        dh = (np.log(grid_max) - np.log(grid_min)) / (n_grid - 1)
        dhdx = 1. / grid
    elif type == 'lin':
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
    
    return dh * weights / dhdx # correct with derivative


def setup_weights():
    pass


