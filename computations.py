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
import scipy.stats


from optimization_linalg import ldp_lawson_hanson, reduce_A_qr, \
    sv_decompose_regularizer, sv_decompose_coeffs
from pycontin_core import RegularizedSolution, IntermediateResults

def _solve_fixed_alpha(inv_input, alpha, intermed_res = None):
    '''
    Inputs:
    inv_input: pycontin_core.InversionInput object
    alpha: magnitude of regularization term
    intermed_res: if provided, use; if not, compute
    '''
    if intermed_res is None: # do initial computations
        # rescale x, A, R, and D
        xsc, alpha_sc, Asc, Rsc = set_scale(inv_input.coeff_matrix,
                                            inv_input.regularizer)
        Dsc = np.dot(inv_input.constraint_matrix, np.diag(xsc))
        # reduce dimensionality of A (Eqn. A.1)
        C, eta = reduce_A_qr(Asc, inv_input.knowns)
        # do svd of regularizer (Eqn. A.7)
        H1_inv, Z = sv_decompose_regularizer(Rsc)
        # svd coefficient matrix (Eqn. A.15)
        CZH1_inv = np.dot(C, np.dot(Z, H1_inv))
        Q, svals, W = sv_decompose_coeffs(CZH1_inv)
        # Eqn. A.19
        gamma = np.dot(Q.transpose(), eta)
        ZH1_invW = np.dot(Z, np.dot(H1_inv, W))
        DZH1_invW = np.dot(Dsc, ZH1_invW)
    else:
        gamma = intermed_res.gamma
        DZH1_invW = intermed_res.DZH1_invW
        ZH1_invW = intermed_res.ZH1_invW
        C = intermed_res.C
        Rsc = intermed_res.Rsc
        Asc = intermed_res.Asc
        svals = intermed_res.svals
        xsc = intermed_res.xsc
        alpha_sc = intermed_res.alpha_sc

    # in subroutine LDPETC, regularizer contribution ||x5||**2 computed before
    # binding constraints are eliminated
    x, binding, success, reg_contrib = setup_and_solve_ldp(alpha, gamma, svals, 
                                                           DZH1_invW, 
                                                           ZH1_invW, 
                                                           inv_input.constraint_rhs)

    # unscale the solution 
    x_unsc = x * xsc

    n_bind = len(binding)

    if n_bind == 0: # no binding constraints
        Gjj = svals / (svals**2 + alpha**2)
        # eqn. A.34
        covar_x = np.dot(ZH1_invW,  
                         np.dot(np.diag(Gjj**2), ZH1_invW.transpose()))
        residuals, chisq, Valpha, n_dof, reduced_chisq = \
            _solution_statistics(Asc, inv_input.knowns, reg_contrib, alpha,
                                 svals, x, n_bind)
    else:
        # extract rows of D and d corresponding to binding constraints
        big_E = inv_input.constraint_matrix[binding] 
        little_e = inv_input.constraint_rhs[binding]
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
        residuals, chisq, Valpha, n_dof, reduced_chisq = \
            _solution_statistics(Asc, inv_input.knowns, reg_contrib, alpha,
                                 new_svals, x, n_bind)
        
    # CONTIN in-line documentation says *dividing* by reduced_chisq
    # which doesn't make sense to me and gives unreasonably huge error bars. 
    # scipy.optimize.linregress does what I'm doing here, and 
    # moreover unit tests agree.
    err_x = np.sqrt(np.diag(covar_x) * reduced_chisq) * xsc

    y_soln = inv_input.knowns - residuals
    solution = RegularizedSolution(x_unsc, err_x, y_soln, reg_contrib,
                                   alpha, binding, residuals,
                                   chisq, Valpha, n_dof, reduced_chisq,
                                   covar_x)
    
    if intermed_res is None:
        intermed_res = IntermediateResults(gamma, DZH1_invW, ZH1_invW,
                                           C, Rsc, Asc, svals, xsc, alpha_sc)
        return solution, intermed_res
    else:
        return solution


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
    #import pdb; pdb.set_trace()
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

    # this depends on alpha
    x, binding, success, reg_contrib = setup_and_solve_ldp(alpha, gamma, svals, 
                                                           DZH1_invW, 
                                                           ZH1_invW, little_d)
    # in subroutine LDPETC, regularizer contribution ||x5||**2 computed before
    # binding constraints are eliminated

    # unscale the solution 
    x_unsc = x * xsc

    n_bind = len(binding)
    #residuals, chisq, V, n_dof, reduced_chisq = \
    #    solution_statistics(Asc, y, alpha, Rsc * alpha_sc / xsc, svals, x, n_bind)
    # TODO: something wrong with the calculation of V(alpha), not sure
    # exactly what


    # put error bars on x
    # if no constraints binding: calc covariance matrix 
    if n_bind == 0: # no binding constraints
        Gjj = svals / (svals**2 + alpha**2)
        # eqn. A.34
        covar_x = np.dot(ZH1_invW,  
                         np.dot(np.diag(Gjj**2), ZH1_invW.transpose()))
        residuals, chisq, V, n_dof, reduced_chisq = \
            _solution_statistics(Asc, y, reg_contrib, alpha,
                                 svals, x, n_bind)
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
        residuals, chisq, V, n_dof, reduced_chisq = \
            _solution_statistics(Asc, y, reg_contrib, alpha, 
                                 new_svals, x, n_bind)
        
    # CONTIN in-line documentation says *dividing* by reduced_chisq
    # which doesn't make sense to me and gives unreasonably huge error bars. 
    # scipy.optimize.linregress  does what I'm doing here, and 
    # moreover unit tests agree.
    err_x = np.sqrt(np.diag(covar_x) * reduced_chisq) * xsc
    
    infodict = {'xsc' : xsc,
                'binding_constraints' : binding,
                'residuals' : residuals,
                'chisq' : chisq,
                'Valpha' : chisq + alpha**2 * reg_contrib,
                'n_dof' : n_dof,
                'reduced_chisq' : reduced_chisq,
                'alpha_sc' : alpha_sc,
                'singular_values': svals,
                'covar_x' : covar_x * reduced_chisq * xsc**2} 
    
    if intermediate_results:
        int_results = {'gamma' : gamma,
                       'DZH1_invW' : DZH1_invW,
                       'ZH1_invW' : ZH1_invW,
                       'C' : C,
                       'Rsc' : Rsc,
                       'Asc' : Asc}
        return x_unsc, err_x, infodict, int_results
    else:
        return x_unsc, err_x, infodict
    

def setup_and_solve_ldp(alpha, gamma, svals, DZH1_invW, ZH1_invW, little_d):
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
    # "regularizer contribution" is norm of x5 in eq. A.17
    regularizer_contrib = np.linalg.norm(np.dot(big_S_tilde_inv, 
                                                (xi + gamma_tilde)))**2
    # Eqn. A.29
    return np.dot(np.dot(ZH1_invW, big_S_tilde_inv), xi + gamma_tilde), \
        binding, success, regularizer_contrib


def _solution_statistics(Asc, y, regularizer_contrib, alpha, svals, x, n_bind):
    # calculate unscaled residuals
    residuals = y - np.dot(Asc, x)
    chisq = np.linalg.norm(residuals)**2

    # calculated V (regularized chisq)
    V = chisq + alpha**2 * regularizer_contrib
    # calculate N_dof, eqns. 3.15 and 3.16
    # sum n_x - n_eq singular values, where n_eq includes binding 
    # inequality constraints that are effectively equality constraints
    n_xe = len(x) - n_bind
    n_dof = np.sum(svals[:n_xe]**2 / (svals[:n_xe]**2 + alpha**2))
    # scaled chi-squared (sigma-hat, eqn. 3.22)
    reduced_chisq = chisq / (len(y) - n_dof)

    return residuals, chisq, V, n_dof, reduced_chisq


def re_solve_fixed_alpha(alpha, gamma, svals, DZH1_invW, ZH1_invW, little_d,
                         xsc, alpha_sc, Rsc, C, Asc, y, big_D):

    x, binding, success, reg_contrib = setup_and_solve_ldp(alpha, gamma, svals,
                                                           DZH1_invW, 
                                                           ZH1_invW, little_d)

    # unscale the solution 
    x_unsc = x * xsc

    n_bind = len(binding)
    
    # put error bars on x
    # if no constraints binding: calc covariance matrix 
    if n_bind == 0: # no binding constraints
        Gjj = svals / (svals**2 + alpha**2)
        # eqn. A.34
        covar_x = np.dot(ZH1_invW,  
                         np.dot(np.diag(Gjj**2), ZH1_invW.transpose()))
        residuals, chisq, V, n_dof, reduced_chisq = \
            _solution_statistics(Asc, y, reg_contrib, alpha,
                                 svals, x, 0)
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
        residuals, chisq, V, n_dof, reduced_chisq = \
            _solution_statistics(Asc, y, reg_contrib, alpha,
                                 new_svals, x, n_bind)

    err_x = np.sqrt(np.diag(covar_x) * reduced_chisq) * xsc
    
    infodict = {'xsc' : xsc,
                'binding_constraints' : binding,
                'residuals' : residuals,
                'chisq' : chisq,
                'Valpha' : chisq + alpha**2 * reg_contrib,
                'n_dof' : n_dof,
                'reduced_chisq' : reduced_chisq,
                'alpha_sc' : alpha_sc,
                'singular_values': svals,
                'covar_x' : covar_x * reduced_chisq * xsc**2} 
    
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
    #print x_scale
    # calculate factor to scale by avg L1 norm of col vectors of A
    factor = (A_col_norms[regularized_vars] * x_scale[regularized_vars]).mean()
    #print factor
    # then scale by average L1 norm of column vectors of A
    x_scale[regularized_vars] = x_scale[regularized_vars] / factor
   
    # scale nonregularized variables by L1 norm of vectors of A
    x_scale[nonregularized_vars] = 1./A_col_norms[nonregularized_vars]

    R_scaled = np.dot(R, np.diag(x_scale))
    # calculate avg element size
    alpha_scale = np.abs(R_scaled).mean()
    R_scaled = R_scaled / alpha_scale
    
    return x_scale, alpha_scale, np.dot(A, np.diag(x_scale)), R_scaled

def prob_1_alpha(V_alpha, V_0, n_dof0, n_data):
    reduced_dof = n_data - n_dof0
    F1_alpha = (V_alpha - V_0)/V_0 * reduced_dof / n_dof0
    return scipy.stats.f.cdf(F1_alpha, n_dof0, reduced_dof)

def solution_series(A, y, R, big_D, little_d, alpha0 = 1e-22, 
                    converge_radius = 0.05):
    # do initial solution with alpha = alpha0
    x0, err_x0, infodict0, int_results = solve_fixed_alpha(A, y, alpha0, R, 
                                                           big_D, little_d,
                                                           True)
    V_0 = infodict0['Valpha']
    ndof_0 = infodict0['n_dof']
    ny = len(y)

    # binary search
    solns = [(x0, err_x0, infodict0, alpha0, 0)]
    prob1_min = V_0
    prob1_max = 0.

    # first search for upper bound
    alpha_trial = alpha0
    while prob1_max < 0.9:
        alpha_trial *= 10
        x, err, infodict = re_solve_fixed_alpha(alpha_trial,
                                                int_results['gamma'],
                                                infodict0['singular_values'],
                                                int_results['DZH1_invW'],
                                                int_results['ZH1_invW'],
                                                little_d,
                                                infodict0['xsc'],
                                                infodict0['alpha_sc'],
                                                int_results['Rsc'],
                                                int_results['C'],
                                                int_results['Asc'],
                                                y, big_D)

        prob1 = prob_1_alpha(infodict['chisq'], V_0, ndof_0, ny)
        solns.append((x, err, infodict, alpha_trial, prob1))
        prob1_max = prob1
        print 'Current alpha: ', alpha_trial
        print 'Current PROB1: ', prob1
        
    # now have lower bound alpha0, upper bound alpha_trial
    working_prob = prob1_max
    alpha_min = alpha0
    alpha_max = alpha_trial

    # go until solution reaches PROB1 = 0.5 +/- converge_radius
    while abs(working_prob - 0.5) > converge_radius:
        alpha_trial = (alpha_max + alpha_min) / 2.
        x, err, infodict = re_solve_fixed_alpha(alpha_trial,
                                                int_results['gamma'],
                                                infodict0['singular_values'],
                                                int_results['DZH1_invW'],
                                                int_results['ZH1_invW'],
                                                little_d,
                                                infodict0['xsc'],
                                                infodict0['alpha_sc'],
                                                int_results['Rsc'],
                                                int_results['C'],
                                                int_results['Asc'],
                                                y, big_D)
        prob1 = prob_1_alpha(infodict['chisq'], V_0, ndof_0, ny)
        solns.append((x, err, infodict, alpha_trial, prob1))
        
        if prob1 >= 0.5:
            alpha_max = alpha_trial
        else:
            alpha_min = alpha_trial

        working_prob = prob1
        print 'Current alpha: ', alpha_trial
        print 'Current PROB1: ', working_prob

    optimal_soln = solns[-1]

    # sort solutions by ascending alpha
    sorted_solns = sorted(solns, key = lambda soln: soln[3])

    return optimal_soln, sorted_solns


def calculate_moments(grid, quad_weights, soln, covariance_matrix = None,
                      moment_range = (-1, 4)):
    '''
    Calculate moments of solution by quadrature.

    grid: grid points
    quad_weights: quadrature weights corresponding to grid points
    soln: 
    soln_errs: error bars on soln
    moment_range: range of moments to calculate

    See p. 237 of Provencher 1982 for discussion of physical
    interpretation of moments.
    '''
    
    def moment_j(j):
        return (quad_weights * grid ** j * soln).sum()

    def error_j(j):
        a0 = quad_weights * grid **j
        return np.sqrt(np.dot(a0, np.dot(covariance_matrix, a0)))

    moments = np.array([moment_j(power) for power in np.arange(*moment_range)])

    if covariance_matrix is not None:
        moment_errors = np.array([error_j(power) for power in 
                                  np.arange(*moment_range)])
        return moments, moment_errors/moments
    else:
        return moments

        
