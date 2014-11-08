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
pycontin_core

Define objects for working with constrained regularized Laplace inversion.

.. moduleauthor:: Jerome Fung <jfung@brandeis.edu>
'''

import numpy as np

from numpy import log, exp
from yaml_serialize import Serializable
#from problem_setup import setup_grid, setup_quadrature
from dls_kernels import molecular_wt_distr, radius_distr

class InversionInput(Serializable):
    def __init__(self, coeff_matrix = None, knowns = None, regularizer = None, 
                 constraint_matrix = None, constraint_rhs = None):
        self.coeff_matrix = coeff_matrix
        self.knowns = knowns
        self.regularizer = regularizer
        self.constraint_matrix = constraint_matrix
        self.constraint_rhs = constraint_rhs


class PyContinInputs(Serializable):
    def __init__(self, n_grid = None, grid_bounds = None, grid_type = 'log',
                 kernel_type = None, kernel_kwargs = None, 
                 weights = None, nonneg = True, dust_term = False,
                 regularizer_type = 'simple', regularizer_external_pts = 2,
                 quadrature_type = 'simpson', external_constraints = None):
        self.n_grid = n_grid
        self.grid_bounds = grid_bounds
        self.grid_type = grid_type
        
        self.kernel_type = kernel_type
        if self.kernel_type == 'mw':
            self.kernel_func = molecular_wt_distr
        elif self.kernel_type == 'rad':
            self.kernel_func = radius_distr
        else:
            raise NotImplementedError

        self.kernel_kwargs = kernel_kwargs
        self.weights = weights
        self.nonneg = nonneg
        self.dust_term = dust_term

        self.regularizer_type = regularizer_type
        self.regularizer_external_pts = regularizer_external_pts

        self.quadrature_type = quadrature_type
        self.external_constraints = external_constraints

    @property
    def grid(self):
        return setup_grid(self.grid_bounds[0], self.grid_bounds[1], 
                          self.n_grid, type = self.grid_type)[0]

    @property
    def grid_props(self):
        dh, dhdx = setup_grid(self.grid_bounds[0], self.grid_bounds[1], 
                              self.n_grid, type = self.grid_type)[1:]
        return dh, dhdx

    @property
    def quad_weights(self):
        return setup_quadrature(self.grid, self.grid_props[0],
                                self.grid_props[1], self.quadrature_type)

class IntermediateResults(Serializable):
    def __init__(self, gamma = None, DZH1_invW = None, ZH1_invW = None, 
                 C = None, Rsc = None, Asc = None, svals = None, 
                 xsc = None, alpha_sc = None):
        self.gamma = gamma
        self.DZH1_invW = DZH1_invW
        self.ZH1_invW = ZH1_invW
        self.C = C
        self.Rsc = Rsc
        self.Asc = Asc
        self.svals = svals
        self.xsc = xsc
        self.alpha_sc = alpha_sc


class RegularizedSolution(Serializable):
    def __init__(self, x = None, error = None, y_soln = None,
                 regularizer_contrib = None, 
                 alpha = None, binding_constraints = None,
                 residuals = None, chisq = None, Valpha = None, n_dof = None,
                 reduced_chisq = None, covariance_matrix = None):
        '''
        intermediate_results: instance of IntermediateResults
        '''
        self.x = x
        self.error = error
        self.y_soln = y_soln
        self.regularizer_contrib = regularizer_contrib
        self.alpha = alpha
        self.binding_constraints = binding_constraints
        self.residuals = residuals
        self.chisq = chisq
        self.Valpha = Valpha
        self.n_dof = n_dof
        self.reduced_chisq = reduced_chisq
        self.covariance_matrix = covariance_matrix


class PhysicalSolution(RegularizedSolution):
    def __init__(self, reg_soln, non_grid_terms = None):
        super(PhysicalSolution, self).__init__(**reg_soln.__dict__)
        self.non_grid_terms = non_grid_terms


class SolutionSeries(Serializable):
    def __init__(self, solutions = None, prob1s = None, optimal_soln = None):
        self.solutions = solutions
        self.prob1s = prob1s
        self.optimal_soln = optimal_soln

# Arguably the following functions belong in problem_setup
# rather than here, but avoid a circular import situation.

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

