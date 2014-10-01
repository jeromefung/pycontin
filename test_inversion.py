'''
'''

import numpy as np
from numpy import sin, cos, exp, pi
from numpy.testing import assert_allclose
import problem_setup
import computations
import scipy.stats

from nose.tools import with_setup

def setup_func():
    # simulate raw dataset: physics
    global tbase, kT, eta, rs, q, data
    tbase = np.logspace(-3, 3, 200) # work in ms, like ALV
    kT = 295 * 1.38e-23
    eta = 1e-3 # Pa s
    a_particle = 1e-7 # meters, 200 nm dia spheres
    theta = pi/2.
    lambda_0 = 633e-9 # HeNe, nm
    n_med = 1.33 # water
    q = 4. * pi * n_med * sin(theta/2.) / lambda_0
    beta = 1.5
    big_Gamma = kT / (6. * pi * eta * a_particle) * q**2
    g1 = 0.5 * exp(-big_Gamma * tbase * 1e-3) + 0.5 * exp(-big_Gamma * 2 * 
                                                           tbase * 1e-3)
    # convert to secs
    data = beta * g1**2

def F_k(rs, tau):
    return exp(-kT / (6 *pi * eta * rs) * q**2 * tau)

def nonneg_sqrt(arr):
    nnsqrt = np.sqrt(np.abs(arr))
    nnsqrt[arr<0] *= -1
    return nnsqrt

# put into form for contin problem
# no dust terms for now
@with_setup(setup_func)
def test_deltafunction():
    n_grid = 51
    grid_r, dh, dhdx = problem_setup.setup_grid(1e-8, 1e-6, n_grid) # log
    quadrature_weights = problem_setup.setup_quadrature(grid_r, dh, dhdx)

    # set up F_k matrix
    matr_Fk = np.zeros((len(tbase), n_grid))

    for i in np.arange(len(tbase)):
        matr_Fk[i] = F_k(grid_r, tbase[i] * 1e-3)

    matrix_A = np.dot(matr_Fk, np.diag(quadrature_weights))

    # set up y
    y_problem = nonneg_sqrt(data)

    # set up nonnegativity constraints
    big_D = np.diag(np.ones(n_grid))
    little_d = np.zeros(n_grid)

    # set up regularizer matrix
    R = problem_setup.setup_regularizer(grid_r, n_grid)

    # how small is too small for alpha?
    # 1e-15 works well
    # 1e-18 is as small as we can go, i think
    # 1e-12 is already perturbing the solution.
    x, err_x, infodict, int_results = computations.solve_fixed_alpha(matrix_A,
                                                                     y_problem,
                                                                     1.1e-21, R,
                                                                     big_D,
                                                                     little_d,
                                                                     True)
                                                                 
    x0, err_x0, infodict0, int_results0 = \
        computations.solve_fixed_alpha(matrix_A, y_problem, 1e-22, R,
                                       big_D, little_d, True)
        
    prob1alpha = computations.prob_1_alpha(infodict['Valpha'],
                                           infodict0['Valpha'],
                                           infodict0['n_dof'],
                                           len(y_problem))

    gold_prob1alpha = 0.64637369
    assert_allclose(prob1alpha, gold_prob1alpha, rtol=1e-6)

    x2, err_x2, infodict2 = \
        computations.re_solve_fixed_alpha(1.1e-21,
                                          int_results['gamma'],
                                          infodict['singular_values'],
                                          int_results['DZH1_invW'],
                                          int_results['ZH1_invW'],
                                          little_d, infodict['xsc'],
                                          infodict['alpha_sc'],  
                                          int_results['Rsc'],
                                          int_results['C'],
                                          int_results['Asc'], y_problem, 
                                          big_D)
    assert_allclose(x, x2)
    assert_allclose(err_x, err_x2)

    best_soln, all_solns = computations.solution_series(matrix_A, y_problem,
                                                        R, big_D, little_d,
                                                        converge_radius=0.02)


@with_setup(setup_func)        
def test_noisy_data():
    n_grid = 51
    grid_r, dh, dhdx = problem_setup.setup_grid(1e-8, 1e-6, n_grid) # log
    quadrature_weights = problem_setup.setup_quadrature(grid_r, dh, dhdx)

    # set up F_k matrix
    matr_Fk = np.zeros((len(tbase), n_grid))

    for i in np.arange(len(tbase)):
        matr_Fk[i] = F_k(grid_r, tbase[i] * 1e-3)

    matrix_A = np.dot(matr_Fk, np.diag(quadrature_weights))

    # set up y
    # see subroutine USERSI
    noise_sigma = 1e-8
    noise = noise_sigma * np.random.randn(len(data))
    noisy_data = data + noise * np.sqrt(data + 1)
    y_problem = nonneg_sqrt(noisy_data)

    # set up nonnegativity constraints
    big_D = np.diag(np.ones(n_grid))
    little_d = np.zeros(n_grid)

    # set up regularizer matrix
    R = problem_setup.setup_regularizer(grid_r, n_grid)

    best_soln, all_solns = computations.solution_series(matrix_A, y_problem,
                                                        R, big_D, little_d,
                                                        converge_radius=0.05)
    print grid_r
    print best_soln

    weights = problem_setup.setup_weights(y_problem - 
                                          best_soln[2]['residuals'])
    weighted_y = weights * y_problem
    weighted_A = np.dot(np.diag(weights), matrix_A)


    weighted_soln, weighted_list = computations.solution_series(weighted_A,
                                                                weighted_y,
                                                                R, big_D,
                                                                little_d,
                                                                converge_radius
                                                                = 0.05)
    #print weighted_soln                 

