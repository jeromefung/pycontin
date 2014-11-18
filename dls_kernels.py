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
dls_kernels

Kernels for Laplace inversion of dynamic light scattering data.

References:
Provencher Comp. Phys. Comm, 1982

"Matrices" and "vectors" refer to 2 and 1-dimensional ndarrays.
'''

import numpy as np
from numpy import pi, exp

def molecular_wt_distr(mw, tau, q = None, prop_const = None):
    '''
    mw : molecular weight
    tau : lag time
    q : scattering wave vector
    prop_const : Proportionality constant such that diffusion constant is 
        D = prop_const / sqrt(mw)

    Units are arbitrary, but should be consistent such that argument
    of exponential is dimensionless.
    '''
    return mw * exp(-prop_const * q**2 * tau / np.sqrt(mw))


def radius_distr(r, tau, q = None, kT = None, eta = None):
    '''
    r : radius
    tau : lag time
    q : scattering wave vector
    kT : absolute temperature in energy units
    eta : solvent viscosity
    '''
    D = kT / (6. * pi * eta * r)
    return r**3 * exp(-D * q**2 * tau)


def diffusion_coeff_distr(D, tau, q = None):
    return exp(-D * q**2 * tau)


def scattering_form_factor(mw, q):
    raise NotImplementedError

