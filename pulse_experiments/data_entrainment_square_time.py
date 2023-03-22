#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from functools import partial
from itertools import product
from neural_field import NeuralField, ParametersBeta, heaviside_firing_rate, exponential_weight_kernel
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2, local_interp
from plotting_helpers import make_animation
from root_finding_helpers import find_roots
from space_domain import SpaceDomain, BufferedSpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper
from tqdm import tqdm

from scipy.stats import linregress
from more_itertools import windowed

import multiprocessing as mp

FILE_NAME = os.path.join(experiment_defaults.data_path,
                         'entrainment_square_time.pickle')

params = ParametersBeta(**{
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0,
})
params_dict = {
        **params.dict,
        'gamma': params.gamma,
        'theta': 0.2,
        'weight_kernel': exponential_weight_kernel
}
xs_right = Domain(0, 200, 8001)
xs_left = Domain(-200, 0, 8001)

"""Finding the speed and pulse width can be slow. Saving them for a given
parameter set helps for rappid testing."""
USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0509375967740198, 9.553535461425781
    print(f'c={c}\nDelta={Delta}')
else:
    Delta_interval = (7, 20)
    speed_interval = (1, 10)
    Delta = find_delta(*Delta_interval, *speed_interval,
                       xs_left, xs_right, verbose=True, **params)
    c = find_c(*speed_interval,  xs_right,
               Delta=Delta, verbose=True, **params)

params_dict['c'] = c
params_dict['Delta'] = Delta

xis, Us, Qs = pulse_profile(xs_right=xs_right, xs_left=xs_left, **params_dict)

space = BufferedSpaceDomain(-100, 200, 10**4, 0.1)
time = TimeDomain_Start_Stop_MaxSpacing(0, 40, 1e-3)

initial_offset = 0
u0 = np.empty((2, space.num_points))
u0[0] = np.array([local_interp(x, xis, Us) for x in space.array])
u0[1] = np.array([local_interp(x, xis, Qs) for x in space.array])

model = NeuralField(
                space=space,
                firing_rate=partial(heaviside_firing_rate,
                                    theta=params_dict['theta']),
                weight_kernel=exponential_weight_kernel,
                params=params)

# solver = TqdmWrapper(Euler())
solver = Euler()
# window_width = 50
stim_start = -2

def check_entrainment(param_tuple):
    stim_speed, stim_magnitude, freq = param_tuple

    def stim_time_modulation(t):
        return np.heaviside(np.sin(2*np.pi*freq*t), 0)

    def stim_func(t):
        return stim_magnitude*np.exp(-np.abs(space.array - stim_start - stim_speed*t)**2)*stim_time_modulation(t)

    def rhs(t, u):
        stim = np.zeros_like(u0)
        stim[0] = stim_func(t)
        return model.rhs(t, u) + stim

    fronts = []
    # front_speeds = []
    entrained = True
    relative_stim_position = None
    for t, (u, q) in zip(time.array, solver.solution_generator(u0, rhs, time)):
        fronts.append(find_roots(space.inner, u[space.inner_slice]-params_dict['theta'], window=3)[-1])
        # if len(fronts) < window_width:
        #     continue
        # front_speed = linregress(time.spacing*np.arange(window_width),
        #                          fronts[-window_width:]).slope
        # if abs(front_speed - stim_speed) < stim_speed/100:
        #     entrained = True
        #     relative_stim_position = fronts[-1] - (stim_speed*t+stim_start)
        #     break

        if fronts[-1] > 0 and (stim_start + stim_speed*t) - fronts[-1] > 3:
            entrained = False
            break

    sol = {
            'stim_speed': stim_speed,
            'stim_magnitude': stim_magnitude,
            'stim_freq': freq,
            'entrained': entrained,
            'relative_stim_position': relative_stim_position
    }
    return sol

results = []
stim_speeds = np.linspace(1.2, 3.5, 21)
# stim_magnitudes = [0.1, 0.3, 0.5] # np.linspace(0, 0.5, 3)
stim_magnitudes = np.linspace(0.1, 0.5, 5)
stim_freqs = np.linspace(0.1, 1.0, 21)

param_list = list(product(stim_speeds, stim_magnitudes, stim_freqs))
with mp.Pool(7) as pool:
    results = list(tqdm(pool.imap(check_entrainment, param_list), total=len(param_list)))


freq_mat, stim_mat = np.meshgrid(stim_freqs, stim_speeds)
fig, axes = plt.subplots(1, len(stim_magnitudes), figsize=(15, 5))
if not hasattr(axes, '__len__'):
    axes = [axes]
for ax, mag in zip(axes, stim_magnitudes):
    res_mat = -np.ones_like(freq_mat, dtype=int)
    for index, (freq, speed) in enumerate(zip(freq_mat.flat, stim_mat.flat)):
        for sol in results:
            if sol['stim_magnitude'] == mag and sol['stim_speed'] == speed and sol['stim_freq'] == freq:
                np.ravel(res_mat)[index] = int(sol['entrained'])

    temp = ax.pcolormesh(freq_mat, stim_mat, res_mat,
                         cmap='seismic', shading='gouraud')
    ax.plot(freq_mat, stim_mat, '.', color='gray')
    ax.set_xlabel('Stimulus frequency')
    ax.set_ylabel('Stimulus Speed')
    ax.set_title(f'stim mag = {mag}')

plt.suptitle('Entrainment to a moving pulsing Gaussian')
# plt.colorbar(temp, ax=axes[-1])
# plt.tight_layout()
plt.show()
plt.savefig('media/entrainment_pulsing.png')

with open(FILE_NAME, 'wb') as f:
    pickle.dump((stim_magnitudes, stim_speeds, stim_freqs, results, params, params_dict), f)
