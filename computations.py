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
computations

Functions for solving constrained optimization problem in CONTIN.

References:
Lawson & Hanson
Provencher Comp. Phys. Comm, 1982

"Matrices" and "vectors" refer to 2 and 1-dimensional ndarrays.
'''

import numpy as np
import numpy.linalg
import scipy.linalg

from scipy.optimize import nnls

def ldp_lawson_hanson(G, h):
    '''
    Finds vector x in R^n of minimum Euclidean norm satisfying
    G x >= h, where G is a m x n real matrix and h is an element of
    R^m.
    
    Implements Lawson & Hanson Algorithm LDP (23.27), p. 165.

    Parameters
    ----------
    G:
        Matrix (m x n) on LHS of inequality constraints
    h:
        Vector (m) on RHS of inequality constraints

    Returns
    -------
    x:
        Vector (n) of optimal solution, if found
    binding_constraints:
        ndarray of indices of elements of x where the inequality 
        constraint is binding
    success:
        Boolean (True) if solution found
    '''
    # dimensions
    m, n = G.shape

    # construct augmented matrix E
    E = np.vstack((G.transpose(), h))
    # construct vector f
    f = np.concatenate((np.zeros(n), np.ones(1)))
    # solve non-negative least squares problem min || E u - f ||
    u, u_residuals = nnls(E, f)

    # compute r
    r = np.dot(E, u) - f

    if u_residuals == 0:
        # constraints are incompatible
        success = False
        return success
    else:
        success = True
        # see discussion of set S on p. 166 and eqn. 23.30 (Kuhn-Tucker)
        binding_constraints = np.where(u[:-1] > 0)[0]
        # compute LDP solution vector
        x = -r[:-1] / r[-1] 
        return x, binding_constraints, success


def reduce_A_qr(A, y):
    '''
    Perform a QR decomposition via Householder transformations
    to reduce coefficient matrix M_e A (N_y x N_x) and data vector 
    Me y to N_x x N_x matrix C and N_x dimensional eta.
   
    See Eqn. A.1 of Provencher 1981.
    '''
    orthog, C = numpy.linalg.qr(A)
    eta = np.dot(orthog.transpose(), y)
    return C, eta


def sv_decompose_regularizer(R):
    '''
    Compute matrices U, H1, and Z in eqn. A.7

    Parameters
    ----------
    R:
        Regularizer matrix, ndarray (n_reg, n_x). If there have been 
        equality constraints, assume they have been eliminated such that
        the input matrix is RK2 (n_reg x n_xe).
    
    Returns
    -------
    H1_inv:
        ndarray (n_x, n_x), diagonal
    Z:
        ndarray (n_x, n_x)
    '''
    n_reg, n_x = R.shape

    # n_reg must be >= n_x
    # add rows of zeros to make n_reg = n_x if needed
    if n_reg < n_x:
        R = np.vstack((R, np.zeros((n_x - n_reg, n_x))))
        n_reg = n_x

    # see svd docstring. H1 is 1d array of singular values
    U, H1, Ztr = np.linalg.svd(R)

    # check if singular values are too close to epsilon
    # increase to fraction of largest SV
    machine_epsilon = np.finfo(np.float64).eps
    abs_singular_vals = np.abs(H1)
    smallest_sv = np.sqrt(machine_epsilon) * abs_singular_vals.max()
    H1 = H1 + (abs_singular_vals < 
               smallest_sv) * (smallest_sv - abs_singular_vals) * np.sign(H1)

    return np.diag(1./H1), Ztr.transpose()


def sv_decompose_coeffs(CK2ZH1inv):
    '''
    eqn a.15
    '''
    Q, S, Wtr = np.linalg.svd(CK2ZH1inv)
    return Q, S, Wtr.transpose()


def solve_fixed_alpha(A, y, alpha, R, big_D, little_d, 
                      intermediate_results = True):
    '''
    Solve regularized inverse problem subject to constraints.
    
    Find x minimizing

    || y - A x ||^2 + alpha^2 * ||R x||^2

    subject to
    
    big_D x >= little_d

    Parameters
    ----------
    A:
        ndarray (n_y, n_x): weighted matrix of coefficients
    y:
        ndarray (n_y): weighted vector of "measurements"
    alpha:
        magnitude of regularizing term
    R:
        regularizing matrix, ndarray (n_reg, n_x)
    big_D:
        LHS matrix of constraint inequality
    little_d
        RHS of constraint inequality
    intermediate_results:
        If True, return DZH1_invW and S matrices to speed soln
        with only a different alpha

    Returns
    -------
    x, weighted residuals, binding constraints
    intermediate matrices: DZH1_invW, S, gamma, xscale
    '''
    # rescale x, A, R, and D
    xsc, alpha_sc, Asc, Rsc = set_scale(A, R)
    Dsc = np.dot(big_D, np.diag(xsc))

    # reduce dimensionality of A (Eqn. A.1)
    C, eta = reduce_A_qr(Asc, y)
 
    # do svd of regularizer (Eqn. A.7)
    H1_inv, Z = sv_decompose_regularizer(Rsc)

    # svd coefficient matrix (Eqn. A.15)
    CZH1_inv = np.dot(C, np.dot(Z, H1_inv))
    Q, svals, W = sv_decompose_coeffs(CZH1_inv)
    
    # Eqn. A.19
    gamma = np.dot(Q.transpose(), eta)
    ZH1_invW = np.dot(Z, np.dot(H1_inv, W))
    DZH1_invW = np.dot(Dsc, ZH1_invW)

    # this part depends on alpha
    # Eqn. A.21 
    gamma_tilde = gamma * svals / np.sqrt(svals**2 + alpha**2)

    # Eqn. A.22
    big_S_tilde_inv = np.diag(1./np.sqrt(svals**2 + alpha**2))
    
    # LHS of ineqeuality A.28
    A28LHS = np.dot(DZH1_invW, big_S_tilde_inv)

    # Eqn. A.28, RHS
    A28RHS = little_d - np.dot(A28LHS, gamma_tilde)

    xi, binding, success = ldp_lawson_hanson(A28LHS, A28RHS)

    # Eqn. A.29
    x = np.dot(np.dot(ZH1_invW, big_S_tilde_inv), xi + gamma_tilde)

    # unscale the solution 
    x_unsc = x * xsc

    # calculate unscaled residuals
    residuals = y - np.dot(Asc, x)
    chisq = np.linalg.norm(residuals)**2

    # calculated V (regularized chisq)
    V = chisq + alpha**2 * alpha_sc * np.linalg.norm(np.dot(Rsc, x))**2

    # calculate N_dof, eqns. 3.15 and 3.16
    n_dof = np.sum(svals**2 / (svals**2 + alpha**2))

    # scaled chi-squared (sigma-hat, eqn. 3.22)
    chisq_scaled = chisq / np.sqrt(len(y) - n_dof)

    # put error bars on x
    # if no constraints binding: calc covariance matrix 
    if len(binding) == 0: # no binding constraints
        Gjj = svals / (svals**2 + alpha**2)
        # eqn. A.34
        covar_x = np.dot(ZH1_invW,  
                         np.dot(np.diag(Gjj**2), ZH1_invW.transpose()))
    else:
        
        # extract rows of D and d corresponding to binding constraints
        big_E = big_D[binding] 
        little_e = little_d[binding]
        K2 = calc_K2(big_E)
        # re-solve problem with K2
        newH1_inv, newZ = sv_decompose_regularizer(np.dot(Rsc, K2))
        newZH1_inv = np.dot(newZ, newH1_inv)
        K2ZH1_inv = np.dot(K2, newZH1_inv)
        CK2ZH1_inv = np.dot(C, K2ZH1_inv)
        newQ, new_svals, newW = sv_decompose_coeffs(CK2ZH1_inv)
        K2ZH1_invW = np.dot(K2ZH1_inv, newW)
        new_Gjj = new_svals / (new_svals**2 + alpha**2)
        covar_x = np.dot(K2ZH1_invW,
                         np.dot(np.diag(new_Gjj**2), K2ZH1_invW.transpose()))

    err_x = np.sqrt(np.diag(covar_x)) * xsc
    
    #err_x = np.zeros(len(x))

    infodict = {'xsc' : xsc,
                'binding_constraints' : binding,
                'residuals' : residuals,
                'chisq' : chisq,
                'Valpha' : V,
                'n_dof' : n_dof,
                'scaled_chisq' : chisq_scaled,
                'alpha_sc' : alpha_sc,
                'singular_values': svals} 
    
    if intermediate_results:
        int_results = {'gamma' : gamma,
                       'DZH1_invW' : DZH1_invW,
                       'ZH1_invW' : ZH1_invW,
                       'C' : C}
        return x_unsc, err_x, infodict, int_results
    else:
        return x_unsc, err_x, infodict
    

def calc_K2(E):
    '''
    Do a QR decomposition to compute the K = [K1 | K2] in Eqn. A.2
    '''
    # E is Neq x Nx, Neq < Nx. Transpose it and QR
    n_eq, n_x = E.shape
    n_xe = n_x - n_eq
    K, r = scipy.linalg.qr(E.transpose())
    return K[:, -n_xe:]

def set_scale(A, R):
    # see subroutine SETSCA
    # calculate L1 norms of columns of A and R
    n_y, n_x = A.shape
    A_col_norms = np.array([np.linalg.norm(A[:,i], ord = 1) 
                            for i in np.arange(n_x)])
    R_col_norms = np.array([np.linalg.norm(R[:,i], ord = 1)
                            for i in np.arange(n_x)])

    regularized_vars = np.nonzero(R_col_norms)
    nonregularized_vars = np.where(R_col_norms == 0)[0]

    x_scale = np.zeros(n_x)
    # scale to normalize L1 norm of column vectors of R
    x_scale[regularized_vars] = 1./R_col_norms[regularized_vars]
    # then scale by average L1 norm of column vectors of A
    x_scale[regularized_vars] = x_scale[regularized_vars] / \
        A_col_norms[regularized_vars].mean() 
   
    # scale nonregularized variables by L1 norm of vectors of A
    x_scale[nonregularized_vars] = 1./A_col_norms[nonregularized_vars]

    R_scaled = np.dot(R, np.diag(x_scale))
    # calculate avg element size
    alpha_scale = np.abs(R_scaled).mean()
    R_scaled = R_scaled / alpha_scale
    
    return x_scale, alpha_scale, np.dot(A, np.diag(x_scale)), R_scaled

