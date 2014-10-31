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
    prob_1_alpha, calculate_moments, re_solve_fixed_alpha, solution_series
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
        self.grid_mw, dh, dhdx = setup_grid(gmnmx[0], gmnmx[1], self.n_grid)
        self.quad_weights = setup_quadrature(self.grid_mw, dh, dhdx)

        # Set up coefficient, constraint, and regularizer matrices
        self.A = problem_setup.setup_coefficient_matrix(self.grid_mw, tbase, 
                                                        molecular_wt_distr,
                                                        mw_dist_kwargs,
                                                        self.quad_weights, 
                                                        True)
        self.big_D, self.little_d = problem_setup.setup_nonneg(self.n_grid + 1)
        self.R = problem_setup.dumb_regularizer(self.grid_mw, self.n_grid + 1,
                                                0)

        # Load matrix stats: table of grid points in CONTIN test output
        self.matrix_stats = np.loadtxt('contin_data_set_1_matrix_stats.txt')
    
        # solve alpha ~ 0 case:
        self.x0, self.err_x0, \
            self.infodict0, self.int_res = solve_fixed_alpha(self.A, self.y,
                                                             5.91e-10, self.R,
                                                             self.big_D,
                                                             self.little_d,
                                                             True)
        # solve regularized case
        self.xa, self.err_xa, \
            self.infodicta = solve_fixed_alpha(self.A, self.y, 3e-6, self.R,
                                               self.big_D, self.little_d, 
                                               False)

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

        assert_allclose(self.xa, gold_x, rtol = 2e-3, atol = 1e-14)
        assert_allclose(self.err_xa, gold_err, rtol = 4e-2, atol = 1e-16)
        assert_allclose(self.infodicta['Valpha'], 3.38453e-4, rtol = 1e-4)
        assert_allclose(self.infodicta['chisq'], 3.15960e-4, rtol = 1e-4)
        assert_allclose(np.sqrt(self.infodicta['reduced_chisq']), 3.056e-3, 
                        rtol = 1e-3)
        assert_allclose(self.infodicta['n_dof'], 3.175, rtol = 1e-3)

        # calculate PROB1 TO REJECT
        prob1 = prob_1_alpha(self.infodicta['chisq'], self.infodict0['Valpha'],
                             self.infodict0['n_dof'], len(self.y))
        assert_allclose(prob1, 0.704, rtol = 1e-3)


    def test_weighted_soln(self):
        # use best-fit solution as weights 
        weights = problem_setup.setup_weights(self.y - 
                                              self.infodicta['residuals'])
        gold_weights = np.array([1.1168, 1.0800, 1.0437, 1.0081, 0.97315,
                                 0.93903, 0.90576, 0.87341, 0.84202, 0.81162,
                                 0.78224, 0.75388, 0.72655, 0.70026, 0.67500,
                                 0.65075, 0.62750, 0.58393, 0.54410, 0.50780,
                                 0.47478, 0.44480, 0.41764, 0.39304, 0.37080,
                                 0.35070, 0.33255, 0.31617, 0.30138, 0.28804,
                                 0.27601, 0.26516, 0.25537, 0.23858, 0.22491,
                                 0.21379, 0.20473])
        assert_allclose(weights, gold_weights, rtol = 1e-4)

        x_wt, err_x_wt, dict_wt = solve_fixed_alpha(np.dot(np.diag(weights), 
                                                           self.A),
                                                    weights * self.y,
                                                    2.23e-7,
                                                    self.R, self.big_D,
                                                    self.little_d, False)
        # see CONTIN test output
        gold_xwt = np.concatenate((np.zeros(18),
                                   np.array([1.014e-11, 3.275e-11, 
                                             1.795e-11]),
                                   np.zeros(10), np.array([8.2845e-2])))
        gold_err = np.concatenate((np.zeros(18),
                                   np.array([4e-12, 2.3e-12, 1.9e-12]),
                                   np.zeros(10), np.array([1.6e-3])))

        assert_allclose(x_wt, gold_xwt, rtol = 2e-3, atol = 1e-15)
        #print err_x_wt
        #print gold_err
        assert_allclose(err_x_wt, gold_err, rtol = 3e-2, atol = 1e-14)
        assert_allclose(dict_wt['n_dof'], 3.038, rtol = 1e-3, atol = 1e-15)
        assert_allclose(np.sqrt(dict_wt['reduced_chisq']), 9.317e-4, 
                        rtol = 1e-3)


    def test_re_solve(self):
        x2, err_x2, infodict2 = re_solve_fixed_alpha(5.91e-10,
                                                     self.int_res['gamma'],
                                                     self.infodict0['singular_values'],
                                                     self.int_res['DZH1_invW'],
                                                     self.int_res['ZH1_invW'],
                                                     self.little_d,
                                                     self.infodict0['xsc'],
                                                     self.infodict0['alpha_sc'],
                                                     self.int_res['Rsc'],
                                                     self.int_res['C'],
                                                     self.int_res['Asc'],
                                                     self.y, self.big_D)
        assert_allclose(x2, self.x0)
        assert_allclose(err_x2, self.err_x0)
        assert_allclose(infodict2['Valpha'], self.infodict0['Valpha'])
        assert_allclose(infodict2['reduced_chisq'], 
                        self.infodict0['reduced_chisq'])
        assert_allclose(infodict2['covar_x'], self.infodict0['covar_x'])

    def test_moment_analysis(self):
        moments, mom_errs = calculate_moments(self.grid_mw, self.quad_weights,
                                              self.x0[:-1], 
                                              self.infodict0['covar_x'][:-1, 
                                                                          :-1])
        # see contin test output
        gold_moments = np.array([1.9509e-11, 3.4570e-6, 6.1951e-1,
                                 1.1257e5, 2.0797e10])
        gold_percent_errs = np.array([2.9, 1.6, 0.28, 2.0, 4.3])
        
        assert_allclose(moments, gold_moments, rtol = 1e-2)
        assert_allclose(mom_errs, 1e-2 * gold_percent_errs, rtol = 2e-2)


    def test_solution_series(self):
        optimal_soln, sorted_solns = solution_series(self.A, self.y,
                                                     self.R, self.big_D,
                                                     self.little_d,
                                                     alpha0=1e-10,
                                                     converge_radius = 0.05)
        assert_allclose(optimal_soln[-1], 0.5, rtol=1e-1)
