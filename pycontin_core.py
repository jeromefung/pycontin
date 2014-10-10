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

from yaml_serialize import Serializable
from problem_setup import setup_grid

class InversionInput(Serializable):
    def __init__(self, coeff_matrix = None, knowns = None, alpha = None, 
                 regularizer = None, constraint_matrix = None, 
                 constraint_rhs = None):
        self.coeff_matrix = coeff_matrix
        self.knowns = knowns
        self.alpha = alpha
        self.regularizer = regularizer
        self.constraint_matrix = constraint_matrix
        self.constraint_rhs = constraint_rhs


class PyContinInputs(Serializable):
    def __init__(self, n_grid = None, grid_bounds = None, grid_type = 'log',
                 kernel_type = None, kernel_kwargs = None, 
                 weights = None, nonneg = True, dust_term = False,
                 regularizer_type = 'simple', regularizer_external_pts = 2,
                 quadrature_type = 'simpson'):
        self.n_grid = n_grid
        self.grid_bounds = grid_bounds
        self.grid_type = grid_type
        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs
        self.weights = weights
        self.nonneg = nonneg
        self.dust_term = dust_term
        self.regularizer_type = regularizer_type
        self.regularizer_external_pts = regularizer_external_pts
        self.quadrature_type = quadrature_type

    @property
    def grid(self):
        return setup_grid(self.grid_bounds[0], self.grid_bounds[1], 
                          self.n_grid, type = self.grid_type)[0]

    @property
    def grid_props(self):
        dh, dhdx = setup_grid(self.grid_bounds[0], self.grid_bounds[1], 
                              self.n_grid, type = self.grid_type)[1:]
        return dh, dhdx


class IntermediateResults(Serializable):
    def __init__(self, gamma = None, ZH1_invW = None, C = None, Rsc = None,
                 Asc = None):
        self.gamma = gamma
        self.ZH1_invW = ZH1_invW
        self.C = C
        self.Rsc = Rsc
        self.Asc = Asc


class RegularizedSolution(Serializable):
    def __init__(self, x = None, error = None, y_soln = None,
                 regularizer_contrib = None, 
                 alpha = None, alpha_sc = None, binding_constraints = None,
                 residuals = None, chisq = None, Valpha = None, n_dof = None,
                 reduced_chisq = None, alpha_sc = None, singular_values = None,
                 covariance_matrix = None, intermediate_results = None):
        '''
        intermediate_results: instance of IntermediateResults
        '''
        self.x = x
        self.error = error
        self.y_soln = y_soln
        self.regularizer_contrib = regularizer_contrib
        self.alpha = alpha
        self.alpha_sc = alpha_sc
        self.binding_constraints = binding_constraints
        self.residuals = residuals
        self.chisq = chisq
        self.Valpha = Valpha
        self.n_dof = n_dof
        self.reduced_chisq = reduced_chisq
        self.singular_values = singular_values
        self.covariance_matrix = covariance_matrix
        self.intermediate_results = intermediate_results


class SolutionSeries(Serializable):
    def __init__(self, solutions = None, prob1s = None, optimal_soln = None):
        self.solutions = solutions
        self.prob1s = prob1s
        self.optimal_soln = optimal_soln

