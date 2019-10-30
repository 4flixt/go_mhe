import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import pickle
import pdb


def sim_LTI(sim_param, A, B, C, D=0, mode=0, seed=None, x0=None, u0=None):
    if seed:
        np.random.seed(seed)

    x_arr = []
    y_arr = []
    u_arr = []
    t_arr = []

    nx = A.shape[0]
    nu = B.shape[1]

    # Initial values:

    if not x0:
        x = np.zeros((nx, 1))
    if not u0:
        u = np.zeros((nu, 1))

    t = 0
    t_reset = np.array([t, t]).reshape(nu, 1)
    stop_phase_counter = 0
    t_stop_phase_init = 0
    while t < sim_param['t_end']:
        if mode == 0:
            # Boolean vector : Should control input be resetted?
            reset = (t-t_reset > sim_param['t_u_min'])*(t-t_reset + np.random.rand(nu, 1)*(
                sim_param['t_u_max']-sim_param['t_u_min']) > sim_param['t_u_max'])
            # Candidate for new control input:
            u_candidate = (-0.5+np.random.rand(nu, 1))*sim_param['u_amp']
            # New control input (if resetted):
            u = (reset*u_candidate+(1-reset)*u)
            # Reset timer (if resetted):
            t_reset = reset*t+(1-reset)*t_reset
        elif mode == 1:
            u1_candidate = sim_param['u_amp']/2*np.sin(t*0.2*(2+np.sin(0.1*t)))
            u2_candidate = sim_param['u_amp']/2*np.cos(t*0.2*(2+np.cos(0.1*t)))
            u = np.stack((u1_candidate, u2_candidate)).reshape(nu, 1)

        # Control input to zero multiple times
        if sim_param['num_stop_phases'] > 0 and stop_phase_counter*sim_param['t_end']/sim_param['num_stop_phases'] <= t:
            u = np.zeros((nu, 1))
            if t-t_stop_phase_init >= sim_param['stop_phase_duration']:
                stop_phase_counter += 1
        else:
            t_stop_phase_init = t

        # Update states (x) and time (t):
        y = np.matmul(C, x)
        x = np.matmul(A, x)+np.matmul(B, u)
        t += sim_param['dt']

        # Save results:
        y_arr.append(y)
        x_arr.append(x)
        u_arr.append(u)
        t_arr.append(t)

    y_arr = np.concatenate(y_arr, axis=1)
    x_arr = np.concatenate(x_arr, axis=1)
    u_arr = np.concatenate(u_arr, axis=1)
    t_arr = np.array(t_arr)

    return y_arr, x_arr, u_arr, t_arr


LTI_system = sio.loadmat('/home/ffiedler/Documents/git_repos/go_mhe/data/LTI_sys_triple_mass_pendulum/LTI_sys_dc')
A, B, C = (LTI_system[key] for key in ['A_dc', 'B_dc', 'C_dc'])
if 'ts' in LTI_system.keys():
    ts = LTI_system['ts']
else:
    ts = 0.075

sim_param = {}
sim_param['dt'] = ts

sim_param['t_end'] = 40
sim_param['t_u_min'] = 0.2  # minimum hold time for control input (avoid rapid movement)
sim_param['t_u_max'] = 2  # maximum hold time for control input
sim_param['u_amp'] = 1*np.pi  # max. amplitude of control input
# control inputs to zero after this time
sim_param['num_stop_phases'] = 3
sim_param['stop_phase_duration'] = 3
mode = 0
y_arr, x_arr, u_arr, t_arr = sim_LTI(sim_param, A, B, C, mode=mode, seed=99)


sio.savemat('001_LTI_results.mat', {'x': x_arr, 'y': y_arr, 'u': u_arr, 't': t_arr})

fig, ax = plt.subplots(2, 1, figsize=(12, 6))

ax[0].plot(t_arr, y_arr.T)
ax[0].set_ylabel('disc angle [rad]')
ax[1].plot(t_arr, u_arr.T)
ax[1].set_ylabel('motor angle [rad]')
ax[1].set_xlabel('time [s]')
fig.align_labels()
fig.tight_layout()
