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
pycontin_fixed_q


.. moduleauthor:: Jerome Fung <jfung@brandeis.edu>
'''

import numpy as np

from pycontin_core import InversionInput, PhysicalSolution, SolutionSeries
from problem_setup import setup_coefficient_matrix, nonneg_sqrt, \
    setup_nonneg, dumb_regularizer, setup_regularizer
from computations import _solve_fixed_alpha, _binary_search
    

def solve_series(measurement, pycontin_inputs, intermed_result = None,
                 alpha_0 = 1e-10, 
                 converge_radius = 0.05, target = 0.5):
    '''
    measurement: instance of dls.core.Measurement
    pycontin_inputs: instance of pycontin_core.PyContinInputs
    
    Returns:
    solution: instance of pycontin_core.SolutionSeries
    '''
    # do some setup here. return an InversionInput object
    inversion_input = _setup_inversion(measurement, pycontin_inputs)
    
    # solve alpha_0 case
    if intermed_result is None:
        soln_0, int_result = _solve_fixed_alpha(inversion_input, alpha_0)
    else:
        soln_0 = _solve_fixed_alpha(inversion_input, alpha_0, intermed_result)
        int_result = intermed_result

    # do binary search
    solns, alphas, prob1s = _binary_search(target, alpha_0, inversion_input,
                                           int_result, soln_0, 
                                           converge_radius)

    if pycontin_inputs.dust_term:
        phys_solns = [PhysicalSolution(soln, ['dust']) for soln in solns]
    else:
        phys_solns = [PhysicalSolution(soln) for soln in solns]


    # now sort solutions
    sort_order = np.argsort(alphas)

    if intermed_result is None:
        return SolutionSeries(sorted(phys_solns, key = lambda sol: sol.alpha),
                              prob1s[sort_order], alphas[sort_order], 
                              phys_solns[-1], prob1s[-1]), int_result
    else:
        return SolutionSeries(sorted(phys_solns, key = lambda sol: sol.alpha),
                              prob1s[sort_order], alphas[sort_order], 
                              phys_solns[-1], prob1s[-1])

    

def solve_alpha(measurement, pycontin_inputs, alpha, intermed_res = None):
    inversion_input = _setup_inversion(measurement, pycontin_inputs)

    if intermed_res is None:
        reg_soln, int_res = _solve_fixed_alpha(inversion_input, alpha)
        if pycontin_inputs.dust_term:
            soln = PhysicalSolution(reg_soln, ['dust'])
        else:
            soln = PhysicalSolution(reg_soln)
        return soln, int_res
    else:
        reg_soln =  _solve_fixed_alpha(inversion_input, alpha, 
                                                intermed_res)
        if pycontin_inputs.dust_term:
            soln = PhysicalSolution(reg_soln, ['dust'])
        else:
            soln = PhysicalSolution(reg_soln)
        return soln


def _setup_inversion(measmnt, pc_inputs):
    '''
    Inputs:
    Measurement object
    PyContinInputs object
    '''
    if pc_inputs.kernel_kwargs is not None:
        kernel_kwargs_with_q = pc_inputs.kernel_kwargs.copy()
    else:
        kernel_kwargs_with_q = {}
    kernel_kwargs_with_q['q'] = measmnt.qsca
    coeff_matrix = setup_coefficient_matrix(pc_inputs.grid,
                                            measmnt.corrfn.delay_times,
                                            pc_inputs.kernel_func,
                                            kernel_kwargs_with_q,
                                            pc_inputs.quad_weights, 
                                            pc_inputs.dust_term)


    knowns = nonneg_sqrt(measmnt.corrfn.data)
    # calculate knowns, applying weights if need be
    if pc_inputs.weights is not None: # weighted analysis
        coeff_matrix = np.dot(np.diag(pc_inputs.weights), coeff_matrix)
        knowns = pc_inputs.weights * knowns

    if pc_inputs.regularizer_type == 'simple':
            regularizer_func = dumb_regularizer
    elif pc_inputs.regularizer_type == 'integral':
        regularizer_func = setup_regularizer
    else:
        raise NotImplementedError
    
    if pc_inputs.dust_term:
        nx = pc_inputs.n_grid + 1
    else:
        nx = pc_inputs.n_grid

    if pc_inputs.regularizer_external_pts <= 2:
        skip_rows = 2 - pc_inputs.regularizer_external_pts
    else:
        raise ValueError('Too many external points specified zero')
    regularizer = regularizer_func(pc_inputs.grid, nx, skip_rows)

    if pc_inputs.nonneg:
        constraint_matrix, constraint_rhs = setup_nonneg(nx)
    elif pc_inputs.external_constraints is not None:
        constraint_matrix, constraint_rhs = pc_inputs.external_constraints
    else:
        constraint_matrix = np.zeros((nx, nx))
        constraint_rhs = np.zeros(nx)

    return InversionInput(coeff_matrix, knowns, regularizer, constraint_matrix,
                          constraint_rhs)
