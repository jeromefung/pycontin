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
import yaml_serialize

from numpy.testing import assert_allclose
from dls_core import Optics, CorrFn, Measurement
from pycontin_core import PyContinInputs
from problem_setup import add_weights
from pycontin_fixed_q import solve_series, solve_alpha, _setup_inversion
from computations import prob_1_alpha


class TestClass():
    def setUp(self):
        # data and metadata using DLS core classes
        optics = Optics(wavelen = 488e-7, index = 1.43) # length in cm
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

        self.soln0, self.int_res0 = solve_alpha(self.measurement, 
                                                self.pc_inputs, 5.91e-10)
        self.soln_r = solve_alpha(self.measurement, self.pc_inputs, 3e-6, 
                                  self.int_res0)
       
        # need to load/define gold here
        self.matrix_stats = np.loadtxt('contin_data_set_1_matrix_stats.txt')

    def test_lowreg_soln(self):
        # check scalings on x and alpha
        assert_allclose(self.int_res0.xsc, self.matrix_stats[:,2], rtol = 5e-4)
        assert_allclose(self.int_res0.alpha_sc, 1./9.302e13, rtol = 1e-4)

        # check solution 
        gold_x = np.concatenate((np.zeros(19),
                                 np.array([4.263e-11, 1.006e-11]),
                                 np.zeros(10),
                                 np.array([8.5963e-2])))
        gold_err = np.concatenate((np.zeros(19),
                                   np.array([3e-12, 3.2e-12]),
                                   np.zeros(10),
                                   np.array([1.7e-3])))
        
        assert_allclose(self.soln0.x, gold_x, rtol = 1e-3, atol = 1e-16)
        assert_allclose(self.soln0.error, gold_err, rtol = 4e-2, atol = 1e-16)

        # check degs of freedom and residuals
        assert_allclose(self.soln0.n_dof, 3.)
        assert_allclose(np.sqrt(self.soln0.reduced_chisq), 2.889e-3, 
                        rtol = 1e-3)


    def test_coefficient_matrix(self):
        # check setup
        inv_input = _setup_inversion(self.measurement, self.pc_inputs)
        matrix_A_maxima = np.array([inv_input.coeff_matrix[:, i].max() for i in
                                    np.arange(self.pc_inputs.n_grid + 1)])
        matrix_A_minima = np.array([inv_input.coeff_matrix[:, i].min() for i in
                                    np.arange(self.pc_inputs.n_grid + 1)])
        assert_allclose(matrix_A_minima, self.matrix_stats[:,0], rtol = 5e-5)
        assert_allclose(matrix_A_maxima, self.matrix_stats[:,1], rtol = 5e-5)

    
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

        assert_allclose(self.soln_r.x, gold_x, rtol = 2e-3, atol = 1e-14)
        assert_allclose(self.soln_r.error, gold_err, rtol = 4e-2, atol = 1e-16)
        assert_allclose(self.soln_r.Valpha, 3.38453e-4, rtol = 1e-4)
        assert_allclose(self.soln_r.chisq, 3.15960e-4, rtol = 1e-4)
        assert_allclose(np.sqrt(self.soln_r.reduced_chisq), 3.056e-3, 
                        rtol = 1e-3)
        assert_allclose(self.soln_r.n_dof, 3.175, rtol = 1e-3)

        prob1 = prob_1_alpha(self.soln_r.chisq, self.soln0.Valpha, 
                             self.soln0.n_dof, 
                             len(self.measurement.corrfn.data))
        assert_allclose(prob1, 0.704, rtol = 1e-3)


    def test_weighted_soln(self):
        soln_wt, intres_wt = solve_alpha(self.measurement,
                                         add_weights(self.pc_inputs, 
                                                     self.soln_r), 2.23e-7)
        # see CONTIN test output
        gold_xwt = np.concatenate((np.zeros(18),
                                   np.array([1.014e-11, 3.275e-11, 
                                             1.795e-11]),
                                   np.zeros(10), np.array([8.2845e-2])))
        gold_err = np.concatenate((np.zeros(18),
                                   np.array([4e-12, 2.3e-12, 1.9e-12]),
                                   np.zeros(10), np.array([1.6e-3])))

        assert_allclose(soln_wt.x, gold_xwt, rtol = 2e-3, atol = 1e-15)
        assert_allclose(soln_wt.error, gold_err, rtol = 3e-2, atol = 1e-14)
        assert_allclose(soln_wt.n_dof, 3.038, rtol = 1e-3, atol = 1e-15)
        assert_allclose(np.sqrt(soln_wt.reduced_chisq), 9.317e-4, 
                        rtol = 1e-3)

    def test_moment_analysis(self):
        '''
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
        pass
        '''

    def test_series(self):
        series, intermed_res = solve_series(self.measurement, self.pc_inputs)
        assert_allclose(series.best_prob1, 0.5, rtol = 1e-1)


    def test_serialization(self):
        pass
