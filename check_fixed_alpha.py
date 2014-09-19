'''
'''

import numpy as np
from numpy import sin, cos, exp, pi

import problem_setup
import computations
import scipy.stats

# simulate raw dataset: physics
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
                                                       tbase * 1e-3)# convert to secs
data = beta * g1**2


def F_k(rs, tau):
    return exp(-kT / (6 *pi * eta * rs) * q**2 * tau)


def nonneg_sqrt(arr):
    nnsqrt = np.sqrt(np.abs(arr))
    nnsqrt[arr<0] *= -1
    return nnsqrt

# put into form for contin problem
# no dust terms for now

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
                                                                 
x0, err_x0, infodict0, int_results0 = computations.solve_fixed_alpha(matrix_A,
                                                                     y_problem,
                                                                     1e-22, R,
                                                                     big_D,
                                                                     little_d,
                                                                     True)
F1_alpha = (infodict['Valpha'] - infodict0['Valpha']) / infodict0['Valpha'] * \
    (len(data) - infodict0['n_dof']) / infodict0['n_dof']


prob1alpha = scipy.stats.f.cdf(F1_alpha, infodict0['n_dof'], 
                               len(data) - infodict0['n_dof'])
print prob1alpha
