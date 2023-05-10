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

FILE_NAME = os.path.join(experiment_defaults.media_path,
                         'entrainment_square_profile')

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
time = TimeDomain_Start_Stop_MaxSpacing(0, 50, 1e-3)

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

stim_speed = params_dict['c'] + 1.5
stim_magnitude = 0.1

def stim_func(t):
    return stim_magnitude*( np.heaviside(-(space.array - stim_start - stim_speed*t), .5)
                           -np.heaviside(-(space.array - stim_start - stim_speed*t) - stim_width, .5))

def rhs(t, u):
    stim = np.zeros_like(u0)
    stim[0] = stim_func(t)
    return model.rhs(t, u) + stim

for t, (u, q) in zip(time.array, solver.solution_generator(u0, rhs, time)):
    pass

front = find_roots(space.inner, u[space.inner_slice]-params_dict['theta'], window=3)[-1]
xs = space.array - front

back = -25.4
stim_front = 3.19

plt.figure(figsize=(5, 3))
plt.plot(xs, u, 'b-', label='Pulse')
plt.plot(xs, stim_func(t), 'm-', label='Stimulus')
plt.plot(xs, params_dict['theta'] + 0*xs, 'k:', label='Threshold')
plt.plot([back, 0], [-.05]*2, 'g.-')
plt.text(back*2/3, -0.12, 'Active Region', color='green')
plt.plot([stim_front-stim_width, stim_front], [-0.19]*2, 'm.-')
plt.text(-stim_width*2/3, -0.28, 'Stimulus Region', color='magenta')
plt.title('Entrainment Profile')
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlim(-40, 20)
plt.ylim(-.3, .8)
plt.tight_layout()
for ext in ['.png', '.eps']:
    plt.savefig(FILE_NAME + ext)
