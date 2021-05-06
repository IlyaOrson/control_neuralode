#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *

import do_mpc


def template_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Certain parameters
    u_m = 0.0572
    u_d = 0.0
    K_N = 393.1
    Y_NX = 504.5
    k_m = 0.00016
    k_d = 0.281
    k_s = 178.9
    k_i = 447.1
    k_sq = 23.51
    k_iq = 800.0
    K_Np = 16.89

    # States struct (optimization variables):
    C_X = model.set_variable('_x', 'C_X')
    C_N = model.set_variable('_x', 'C_N')
    C_qc = model.set_variable('_x', 'C_qc')

    # Input struct (optimization variables):
    I = model.set_variable('_u', 'I')
    F_N = model.set_variable('_u', 'F_N')

    # # Uncertain parameters:
    # Y_x = model.set_variable('_p',  'Y_x')
    # S_in = model.set_variable('_p', 'S_in')

    # auxiliary variables
    I_ksi = I/(I+k_s+I**2/k_i)
    CN_KN = C_N/(C_N+K_N)

    I_kiq = I/(I+k_sq+I**2/k_iq)
    Cqc_KNp = C_qc/(C_N+K_Np)

    # Differential equations
    model.set_rhs('C_X', u_m * I_ksi * C_X * CN_KN - u_d * C_X)
    model.set_rhs('C_N', -Y_NX * u_m * I_ksi * C_X * CN_KN + F_N)
    model.set_rhs('C_qc', k_m * I_kiq * C_X - k_d * Cqc_KNp)

    # Build the model
    model.setup()

    return model
