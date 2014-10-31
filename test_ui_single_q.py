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

import numpy as np

from dls_core import Optics, CorrFn, Measurement

class TestClass():
    def setUp(self):
        # data and metadata using DLS core classes
        optics = Optics(wavelen = 488., index = 1.33)
        test_data = np.loadtxt('contin_test_data_set_1.txt')
        corr_func = CorrFn(*test_data.transpose())
        self.measurement = Measurement(corr_func, 60., optics)

        # pycontin inputs
        self.pc_inputs = PyContinInputs(n_grid = 31, 
                                        grid_bounds = np.array([5e2, 5e6]),
                                        kernel_type = 'mw',
                                        kernel_kwargs = {'prop_const' : 
                                                         1.37e-4},
                                        dust_term = True)

        # need to load/define gold here

    def test_series(self):
        series = pycontin_fixed_q.solve_series(self.measurement, 
                                               self.pc_inputs)
    
