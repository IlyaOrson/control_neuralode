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

import time

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pandas as pd

import do_mpc

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = True
store_results = True

"""
Get configured do-mpc modules:
"""

horizon = 60
t_step = 4.0
model = template_model()
mpc = template_mpc(model, horizon, t_step)
simulator = template_simulator(model, t_step)
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
Run MPC main loop:
"""
mpc_evol = {
    "C_X": np.zeros(horizon),
    "C_N": np.zeros(horizon),
    "C_qc": np.zeros(horizon),
    "I": np.zeros(horizon),
    "F_N": np.zeros(horizon),
}
states = np.zeros((3,horizon))
controls = np.zeros((2,horizon))
mpc_prediction = {}
for k in range(horizon):
    mpc = template_mpc(model, horizon, t_step)
    mpc.x0 = x0
    try:
        mpc.u0 = u0
    except:
        pass
    mpc.set_initial_guess()

    mpc_evol["C_X"][k], mpc_evol["C_N"][k], mpc_evol["C_qc"][k] = x0.flatten()

    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    mpc_evol["I"][k], mpc_evol["F_N"][k] = u0.flatten()

    if k == 0:
        # removes last one because it is out of range
        mpc_prediction["C_X"] = mpc.data.prediction(("_x", "C_X", 0)).flatten()[:-1]
        mpc_prediction["C_N"] = mpc.data.prediction(("_x", "C_N", 0)).flatten()[:-1]
        mpc_prediction["C_qc"] = mpc.data.prediction(("_x", "C_qc", 0)).flatten()[:-1]

        mpc_prediction["I"] = mpc.data.prediction(("_u", "I", 0)).flatten()
        mpc_prediction["F_N"] = mpc.data.prediction(("_u", "F_N", 0)).flatten()

    # sometimes the prediction is completely off...
    # pd.DataFrame(mpc_prediction).plot(subplots=True, sharex=True); plt.show()

    horizon -= 1


input('Press any key to exit.')

# Store results:
if store_results:
    # do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')
    time_array = [i * mpc.t_step for i in range(len(mpc_evol["I"]))]
    pd.DataFrame(mpc_prediction, index=time_array).to_csv("predictions.csv", index=False)
    pd.DataFrame(mpc_evol, index=time_array).to_csv("evolution.csv", index=False)
