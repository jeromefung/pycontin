import numpy as np
from numpy import sin, cos, exp, pi
from numpy.testing import assert_allclose
import problem_setup
import computations
import scipy.stats

# physical parameters
lambda_0 = 488. # nm
n_med = 1.43
theta_deg = 60.
theta = theta_deg * pi / 180.
q = 4. * pi * n_med * sin(theta / 2.) / lambda_0 * 1e7 # convert to cm^-1
prop_const = 1.37e-4

def F_k(mw, tau):
    return mw * exp(-prop_const * q**2 * tau / np.sqrt(mw))

test_data = np.loadtxt('contin_test_data_set_1.txt')
#print test_data
tbase = test_data[:,0]
y = problem_setup.nonneg_sqrt(test_data[:,1])
#print np.sqrt(test_data[:,1])
#print y
#y = test_data[:,1].copy()

# solution grid
n_grid = 31
gmnmx = [5e2, 5e6]
grid_mw, dh, dhdx = problem_setup.setup_grid(gmnmx[0], gmnmx[1], n_grid)
quadrature_weights = problem_setup.setup_quadrature(grid_mw, dh, dhdx)
#print grid_mw

# F_k matrix
Fk = np.zeros((len(tbase), n_grid))
for i in np.arange(len(tbase)):
    Fk[i] = F_k(grid_mw, tbase[i])

# add a column to A to include a dust term. sign is negative.
# or is it????
matrix_A = np.zeros((len(tbase), n_grid + 1))
matrix_A[:, :-1] = np.dot(Fk, np.diag(quadrature_weights))
matrix_A[:, -1] = np.ones(len(tbase))

# nonnegativity constraints on solution and on dust term
big_D = np.diag(np.ones(n_grid + 1))
little_d = np.zeros(n_grid + 1)

# regularizer, extra column of zeros added
R = problem_setup.setup_regularizer(grid_mw, n_grid + 1)

#print R

# preliminary unweighted analysis
best_uw, solns_uw = computations.solution_series(matrix_A, y, R, big_D,
                                                 little_d, alpha0=5.91e-20)

Ax = np.dot(matrix_A, solns_uw[0][0])

# calculate weights
#weights = problem_setup.setup_weights(y - best_uw[2]['residuals'])
#weighted_y = weights * y
#weighted_A = np.dot(np.diag(weights), matrix_A)

#best_w, solns_w = computations.solution_series(weighted_a, weighted_y, R,
#                                               big_D, little_d)

test_x = np.zeros(n_grid + 1)
test_x[19] = 4.263e-11
test_x[20] = 1.006e-11
test_x[-1] = 8.5963e-2
Ax_test = np.dot(matrix_A, test_x)

test_x_2 = np.zeros(n_grid + 1)
test_x_2[15] = 6.270e-14
test_x_2[16] = 4.878e-12
test_x_2[17] = 1.318e-11
test_x_2[18] = 2.107e-11
test_x_2[19] = 2.259e-11
test_x_2[20] = 1.381e-11
test_x_2[21] = 2.055e-12
test_x_2[-1] = 8.229e-2

# is Ax_test normalized??
print quadrature_weights
print quadrature_weights.sum()
print 'integral of test soln:', (quadrature_weights * test_x[:-1]).sum()
print 'integral of test soln2:', (quadrature_weights * test_x_2[:-1]).sum()
