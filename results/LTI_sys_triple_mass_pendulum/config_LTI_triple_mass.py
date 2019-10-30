import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import time
import pdb
from casadi import *

# Add "code" libraries to path.
import sys
sys.path.append('../../code/')

import go_mhe


gmhe_sim = go_mhe.simulator(n_horizon=20)


gmhe_sim.t_0 = 0                 # inital time step
gmhe_sim.t_step = 0.075          # time step

"""
--------------------------------------------------------------------------
data: Load data
--------------------------------------------------------------------------
"""
dataset = '../../data/LTI_sys_triple_mass_pendulum/001_LTI_results.mat'


gmhe_sim.sim_data = sio.loadmat(dataset)

# Noise and offset parameters:
gmhe_sim.u_noise_mag = 1e-1
gmhe_sim.u_bias = 3

gmhe_sim.y_noise_mag = 2e-1

"""
--------------------------------------------------------------------------
MHE: Tuning
--------------------------------------------------------------------------
"""
# Arrival cost (states)
gmhe_sim.obj_p_num['p_set', 'P_x'] = np.eye(8)

# Arrival cost (parameters)
gmhe_sim.obj_p_num['p_set', 'P_p'] = np.eye(2)

# MHE tuning matrix for input penalty
gmhe_sim.obj_p_num['p_set', 'P_u'] = np.eye(2)

# MHE tuning matrix for measurement penalty
gmhe_sim.obj_p_num['p_set', 'P_y'] = np.eye(3)


"""
--------------------------------------------------------------------------
MHE: Initial guess and estimate
--------------------------------------------------------------------------
"""

gmhe_sim.obj_x_num['x', :] = np.zeros((8, 1))

gmhe_sim.obj_x_num['u', :, 'pos_mot_1'] = 0
gmhe_sim.obj_x_num['u', :, 'pos_mot_2'] = 0

# Initial values for estimated parameters:
gmhe_sim.obj_x_num['p_est', 'mot_1_offset'] = 0
gmhe_sim.obj_x_num['p_est', 'mot_2_offset'] = 0


gmhe_sim.obj_p_num['x_0'] = gmhe_sim.obj_x_num['x', 0]
gmhe_sim.obj_p_num['p_0'] = gmhe_sim.obj_x_num['p_est']


counter_max = gmhe_sim.sim_data['t'].shape[1]  # 12100

time_iter = []

for k in range(counter_max):
    tic = time.time()
    gmhe_sim.mhe_step()
    toc = time.time()
    time_iter.append(toc-tic)
    if np.mod(k, 100) == 0:
        try:
            gmhe_sim.mhe_data.export_to_matlab()
            gmhe_sim.mhe_data.export_cas_struct()
        except:
            print('Couldnt export data.')

gmhe_sim.mhe_data.export_to_matlab()
gmhe_sim.mhe_data.export_cas_struct()

time_iter = np.array(time_iter)
print('Mean time per iteration (with optimization): {} s'.format(time_iter[20:].mean().round(3)))
print('Total time for execution (w/o time for storing solution): {} s'.format(time_iter.sum().round(1)))
