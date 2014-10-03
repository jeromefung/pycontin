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
Unit tests using "TEST DATA SET 1 (MOLECULAR WEIGHT DISTRIBUTION)"
distributed with original Fortran CONTIN.
'''

import numpy as np
import problem_setup
import dls_kernels

from numpy.testing import assert_allclose
from computations import ldp_lawson_hanson, reduce_A_qr, solve_fixed_alpha
from problem_setup import setup_grid, setup_quadrature
from dls_kernels import molecular_wt_distr

from numpy import pi, sin

class TestClass():
    def setUp(self):
        # Physical parameters
        lambda_0 = 488. # nm
        n_med = 1.43
        theta_deg = 60.
        theta = theta_deg * pi / 180.
        q = 4. * pi * n_med * sin(theta / 2.) / lambda_0 * 1e7 
        # convert to cm^-1
        prop_const = 1.37e-4
        mw_dist_kwargs = {'q' : q, 'prop_const' : prop_const}

        # Load/preprocess data
        test_data = np.loadtxt('contin_test_data_set_1.txt')
        tbase = test_data[:,0]
        self.y = problem_setup.nonneg_sqrt(test_data[:,1])

       
        # Solution setup
        self.n_grid = 31
        gmnmx = np.array([5e2, 5e6]) # bounds of grid
        grid_mw, dh, dhdx = setup_grid(gmnmx[0], gmnmx[1], self.n_grid)
        quadrature_weights = setup_quadrature(grid_mw, dh, dhdx)

        # Set up coefficient, constraint, and regularizer matrices
        self.A = problem_setup.setup_coefficient_matrix(grid_mw, tbase, 
                                                        molecular_wt_distr,
                                                        mw_dist_kwargs,
                                                        quadrature_weights, 
                                                        True)
        self.big_D, self.little_d = problem_setup.setup_nonneg(self.n_grid + 1)
        self.R = problem_setup.dumb_regularizer(grid_mw, self.n_grid + 1, 0)

        # Load matrix stats
        self.matrix_stats = np.loadtxt('contin_data_set_1_matrix_stats.txt')
    
    def tearDown(self):
        pass
    
    def test_coefficient_matrix(self):
        

        matrix_A_maxima = np.array([self.A[:,i].max() for i in 
                                    np.arange(self.n_grid + 1)])
        matrix_A_minima = np.array([self.A[:,i].min() for i in 
                                    np.arange(self.n_grid + 1)])
        
        assert_allclose(matrix_A_minima, self.matrix_stats[:,0], rtol = 5e-5)
        assert_allclose(matrix_A_maxima, self.matrix_stats[:,1], rtol = 5e-5)

    def test_solution(self):
        x, err_x, infodict = solve_fixed_alpha(self.A, self.y, 1e-10, 
                                               self.R, self.big_D, 
                                               self.little_d, False)
        assert_allclose(infodict['xsc'], self.matrix_stats[:,2], rtol = 5e-4)
        assert_allclose(infodict['alpha_sc'], 1./9.302e13, rtol = 1e-4)
