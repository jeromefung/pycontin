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
from computations import ldp_lawson_hanson, reduce_A_qr, solve_fixed_alpha, \
    prob_1_alpha
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

        # Load matrix stats: table of grid points in CONTIN test output
        self.matrix_stats = np.loadtxt('contin_data_set_1_matrix_stats.txt')
    
        # solve alpha ~ 0 case:
        (self.x0, self.err_x0, 
         self.infodict0, self.int_res) = solve_fixed_alpha(self.A, self.y,
                                                           5.91e-10, self.R,
                                                           self.big_D,
                                                           self.little_d,
                                                           True)
    def tearDown(self):
        pass
    
    def test_coefficient_matrix(self):
        

        matrix_A_maxima = np.array([self.A[:,i].max() for i in 
                                    np.arange(self.n_grid + 1)])
        matrix_A_minima = np.array([self.A[:,i].min() for i in 
                                    np.arange(self.n_grid + 1)])
        
        assert_allclose(matrix_A_minima, self.matrix_stats[:,0], rtol = 5e-5)
        assert_allclose(matrix_A_maxima, self.matrix_stats[:,1], rtol = 5e-5)


    def test_lowreg_solution(self):
        # check scalings on x and alpha
        assert_allclose(self.infodict0['xsc'], self.matrix_stats[:,2], 
                        rtol = 5e-4)
        assert_allclose(self.infodict0['alpha_sc'], 1./9.302e13, rtol = 1e-4)

        # check solution 
        gold_x = np.concatenate((np.zeros(19),
                                 np.array([4.263e-11, 1.006e-11]),
                                 np.zeros(10),
                                 np.array([8.5963e-2])))
        gold_err = np.concatenate((np.zeros(19),
                                   np.array([3e-12, 3.2e-12]),
                                   np.zeros(10),
                                   np.array([1.7e-3])))

        
        assert_allclose(self.x0, gold_x, rtol = 1e-3, atol = 1e-16)
        assert_allclose(self.err_x0, gold_err, rtol = 4e-2, atol = 1e-16)

        # check degs of freedom and residuals
        assert_allclose(self.infodict0['n_dof'], 3.)
        assert_allclose(np.sqrt(self.infodict0['reduced_chisq']), 2.889e-3, 
                        rtol = 1e-3)
        assert_allclose(self.infodict0['Valpha'], 2.838e-4, rtol = 1e-3)


    def test_reg_soln(self):
        x, err_x, infodict = solve_fixed_alpha(self.A, self.y, 3e-6, 
                                               self.R, self.big_D, 
                                               self.little_d, False)

        gold_x = np.concatenate((np.zeros(15),
                                 np.array([6.270e-14, 4.878e-12,
                                           1.318e-11, 2.107e-11,
                                           2.259e-11, 1.381e-11,
                                           2.055e-12]),
                                 np.zeros(9),
                                 np.array([8.22e-2])))
        gold_err = np.concatenate((np.zeros(15),
                                   np.array([1.9e-12, 2.7e-12, 2e-12,
                                             8.8e-13, 1.7e-12,
                                             9.5e-13, 6e-13]),
                                   np.zeros(9),
                                   np.array([1.9e-3])))

        assert_allclose(x, gold_x, rtol = 2e-3, atol = 1e-14)
        assert_allclose(err_x, gold_err, rtol = 4e-2, atol = 1e-16)
        assert_allclose(infodict['Valpha'], 3.38453e-4, rtol = 1e-4)
        assert_allclose(infodict['chisq'], 3.15960e-4, rtol = 1e-4)
        assert_allclose(np.sqrt(infodict['reduced_chisq']), 3.056e-3, 
                        rtol = 1e-3)
        assert_allclose(infodict['n_dof'], 3.175, rtol = 1e-3)

        # calculate PROB1 TO REJECT
        prob1 = prob_1_alpha(infodict['chisq'], self.infodict0['Valpha'],
                             self.infodict0['n_dof'], len(self.y))
        print 'PROB 1 TO REJECT', prob1
        assert_allclose(prob1, 0.704, rtol = 1e-3)


    def test_resolve(self):
        pass
