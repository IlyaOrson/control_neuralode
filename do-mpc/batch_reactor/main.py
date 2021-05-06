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
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *

import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

C_X_s_0 = 1.0
C_N_s_0 = 150.0
C_qc_s_0 = 0.0
x0 = np.array([C_X_s_0, C_N_s_0, C_qc_s_0])


mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(8,5))
plt.ion()

"""
Run MPC main loop:
"""

# TODO: should only solve until fixed final time by
#       reducing horizon as it advances accordingly
for k in range(120):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)


    # mpc.set_param(n_horizon=mpc.n_horizon-2)
    mpc.n_horizon -= 2
    print(
        "------------------",
        mpc.n_horizon,
        "------------------",
    )

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')
