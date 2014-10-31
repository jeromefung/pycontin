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
        
                # need to load/define gold here

    def test_series(self):
        series = pycontin_fixed_q.solve_multiq()

    
