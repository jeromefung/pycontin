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

from pycontin_core import InversionInput
from problem_setup import setup_coefficient_matrix, nonneg_sqrt, \
    setup_nonneg, dumb_regularizer, setup_regularizer
from computations import _solve_fixed_alpha
    

def solve_series(measurement, pycontin_inputs, alpha_0 = 1e-10):
    '''
    measurement: instance of dls.core.Measurement
    pycontin_inputs: instance of pycontin_core.PyContinInputs
    
    Returns:
    solution: instance of pycontin_core.SolutionSeries
    '''
    # do some setup here. return an InversionInput object
    inversion_input = _setup_inversion(measurement, pycontin_inputs)
    

def solve_alpha(measurement, pycontin_inputs, alpha, intermed_res = None):
    inversion_input = _setup_inversion(measurement, pycontin_inputs)

    if intermed_res is None:
        return _solve_fixed_alpha(inversion_input, alpha)
    else:
        return _solve_fixed_alpha(inversion_input, alpha, intermed_res)

        
def _setup_inversion(measmnt, pc_inputs):
    '''
    Inputs:
    Measurement object
    PyContinInputs object
    '''
    kernel_kwargs_with_q = pc_inputs.kernel_kwargs.copy()
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
    
    if pc_inputs.dust_term is not None:
        nx = pc_inputs.n_grid + 1
    else:
        nx = pc_inputs.ngrid

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
