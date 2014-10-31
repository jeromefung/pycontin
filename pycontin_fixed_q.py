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

from pycontin_core import *

def solve_series(measurement, pycontin_inputs, alpha_0 = 1e-10):
    '''
    measurement: instance of dls.core.Measurement
    pycontin_inputs: instance of pycontin_core.PyContinInputs
    
    Returns:
    solution: instance of pycontin_core.SolutionSeries
    '''
    # do some setup here. return an InversionInput object
    inversion_input = setup_inversion(measurement, pycontin_inputs)
    pass


def solve_alpha(measurement, pycontin_inputs, alpha, reference_soln = None):
    # does solve_alpha need a reference solution??
    pass
