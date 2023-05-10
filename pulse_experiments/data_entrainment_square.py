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
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2, local_interp, local_diff
from plotting_helpers import make_animation
from root_finding_helpers import find_roots
from space_domain import SpaceDomain, BufferedSpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper
from tqdm import tqdm

from scipy.stats import linregress
from more_itertools import windowed

FILE_NAME = os.path.join(experiment_defaults.data_path,
                         'entrainment_square.pickle')

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
time = TimeDomain_Start_Stop_MaxSpacing(0, 100, 1e-3)

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

solver = TqdmWrapper(Euler())
window_width = 10

stim_start = 0
stim_width = 10

# compute approximation
xs = space.array
domain = Domain(xs[0], xs[-1], len(xs))
dU = np.array([local_diff(z, xis, Us, width=2) for z in space.array])
dQ = np.array([local_diff(z, xis, Qs, width=2) for z in space.array])
A0, AmD = nullspace_amplitudes(xis, Us, Qs, **params_dict)
v1_arr = v1(space.array, A0=A0, AmD=AmD, **params_dict)
v2_arr = v2(space.array, A0=A0, AmD=AmD, **params_dict)

denom = -domain.quad(params.mu*dU*v1_arr + params.alpha*dQ*v2_arr)
slope = params_dict['c'] * params_dict['mu']/denom


results = []
# stim_speeds = np.linspace(1.2, 5, 21)
# stim_magnitudes = np.linspace(0, 0.5, 21)
stim_speeds = params_dict['c'] + np.linspace(0.0, 1.5, 21)
stim_magnitudes = np.linspace(0, 0.1, 21)

for stim_speed, stim_magnitude in tqdm(list(product(stim_speeds, stim_magnitudes))):
    def stim_func(t):
        return stim_magnitude*( np.heaviside(-(space.array - stim_start - stim_speed*t), .5)
                               -np.heaviside(-(space.array - stim_start - stim_speed*t) - stim_width, .5))

    def rhs(t, u):
        stim = np.zeros_like(u0)
        stim[0] = stim_func(t)
        return model.rhs(t, u) + stim

    fronts = []
    front_speeds = []
    entrained = False
    relative_stim_position = None
    for t, (u, q) in zip(time.array, solver.solution_generator(u0, rhs, time)):
        fronts.append(find_roots(space.inner, u[space.inner_slice]-params_dict['theta'], window=3)[-1])
        if len(fronts) < window_width:
            continue
        front_speed = linregress(time.spacing*np.arange(window_width),
                                 fronts[-window_width:]).slope
        # if stim_speed*t+stim_start - stim_width > fronts[-1]:
        #     entrained = False
        #     relative_stim_position = fronts[-1] - (stim_speed*t+stim_start)
        #     break
        if abs(front_speed - stim_speed) < stim_speed/100:
            entrained = True
            relative_stim_position = fronts[-1] - (stim_speed*t+stim_start)
            break

    sol = {
            'stim_speed': stim_speed,
            'stim_magnitude': stim_magnitude,
            'entrained': entrained,
            'relative_stim_position': relative_stim_position
    }
    results.append(sol)


plt.xlim(np.min(stim_magnitudes), np.max(stim_magnitudes))
plt.ylim(np.min(stim_speeds), np.max(stim_speeds))

mag_mat, stim_mat = np.meshgrid(stim_magnitudes, stim_speeds)

# z_max = np.floor(np.max(np.nan_to_num(res, nan=-np.inf)))
# z_min = np.ceil(np.min(np.nan_to_num(res, nan=np.inf)))
# z_max, z_min = max(z_max, -z_min), min(-z_max, z_min)

res_mat = np.zeros_like(mag_mat, dtype=bool)
for index, (mag, speed) in enumerate(zip(mag_mat.flat, stim_mat.flat)):
    for sol in results:
        if sol['stim_magnitude'] == mag and sol['stim_speed'] == speed:
            np.ravel(res_mat)[index] = sol['entrained']
            break

plt.pcolormesh(mag_mat, stim_mat, res_mat,
               cmap='seismic', shading='gouraud')
plt.plot(mag_mat, stim_mat, '.', color='gray')
# plt.colorbar(label='$\\nu_\\infty$')
plt.plot(stim_magnitudes, params_dict['c']+stim_magnitudes*slope*(1-np.exp(-stim_width/c)), 'w-', label='Asymptotic')
plt.xlabel('Stimulus Magnitude')
plt.ylabel('Stimulus Speed')
plt.title('Entrainment to a moving square wave')
plt.show()

with open(FILE_NAME, 'wb') as f:
    pickle.dump((stim_magnitudes, stim_speeds, results, stim_width, slope, params, params_dict), f)
