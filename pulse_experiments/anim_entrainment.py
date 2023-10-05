#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os.path
from functools import partial
from itertools import islice
from scipy.stats import linregress

from neural_field_synaptic_depression.neural_field import (
        NeuralField,
        ParametersBeta,
        heaviside_firing_rate,
        exponential_weight_kernel,
)
from neural_field_synaptic_depression.root_finding_helpers import find_roots
from neural_field_synaptic_depression.space_domain import (
        SpaceDomain,
        BufferedSpaceDomain,
)
from neural_field_synaptic_depression.time_domain import (
        TimeDomain,
        TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler, EulerDelta
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper
from plotting_helpers.plotting_helpers import make_animation
from num_assist import (
        Domain,
        find_delta,
        find_c,
        pulse_profile,
        nullspace_amplitudes,
        v1,
        v2,
        local_interp,
)


FILE_NAME = os.path.join(experiment_defaults.media_path,
                         'entrainment_test.gif')

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
time = TimeDomain_Start_Stop_MaxSpacing(0, 15, 1e-3)

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

# FILE_NAME = 'entrainment1_entrainment.gif'
stim_speed = 2.0
stim_start = -5
stim_magnitude = .2

# FILE_NAME = 'entrainment2_entrainment_failure'
# stim_speed = 2.0
# stim_start = -2
# stim_magnitude = .1

# FILE_NAME = 'entrainment3_entrainment'
# stim_speed = 1.5
# stim_start = -5
# stim_magnitude = 0.1

def stim_func(t):
    return stim_magnitude*np.exp(-np.abs(space.array - stim_start - stim_speed*t)**2)

def rhs(t, u):
    stim = np.zeros_like(u0)
    stim[0] = stim_func(t)
    return model.rhs(t, u) + stim


fig, axes = plt.subplots(2, 1, figsize=(10, 10))
u_line, = axes[0].plot(space.array, u0[0], 'b-', label='activity')
q_line, = axes[0].plot(space.array, u0[1], 'b--', label='synaptic efficacy')
stim_line, = axes[0].plot(space.array, stim_func(0), 'm:', label='stimulus')
front_line, = axes[0].plot([0], [params_dict['theta']], 'k.')
stim_speed_line, = axes[0].plot([stim_start], [params_dict['theta']], 'k.')
axes[0].set_xlim(-20, 100)
axes[0].set_xlabel('$x$')
axes[0].legend(loc='upper left')
axes[1].plot([], [], 'b.', label='front')
axes[1].plot(time.array, stim_speed + 0*time.array, 'k-', label='stim')
axes[1].plot(time.array, c + 0*time.array, 'b-', label='natural')
axes[1].set_ylabel('speed')
axes[1].legend()
axes[1].set_xlabel('time')
fronts = []
steps_per_frame = 97
window_width = 10
with imageio.get_writer(FILE_NAME, mode='I') as writer:
    for index, (t, (u, q)) in enumerate(zip(time.array, solver.solution_generator(u0, rhs, time))):
        if index % steps_per_frame != 0:
            continue
        u_line.set_ydata(u)
        q_line.set_ydata(q)
        stim_line.set_ydata(stim_func(t))
        fronts.append(find_roots(space.inner, u[space.inner_slice]-params_dict['theta'], window=3)[-1])
        front_line.set_xdata([fronts[-1]])
        if len(fronts) > window_width:
            # front_speed = (fronts[-1] - fronts[-2])/(time.spacing*steps_per_frame)
            front_speed = linregress([time.spacing*steps_per_frame]*np.arange(window_width),
                                     fronts[-window_width:]).slope
            axes[1].plot(t, front_speed, 'b.', label='front')

        stim_speed_line.set_xdata([stim_speed*t + stim_start])
        plt.savefig(FILE_NAME + '.png')
        image = imageio.imread(FILE_NAME + '.png')
        writer.append_data(image)
        # plt.pause(0.001)

# plt.savefig('media/entrainment_test.png')

fronts = []
for t, (u, q) in zip(time.array, solver.solution_generator(u0, rhs, time)):
    fronts.append(find_roots(space.inner, u[space.inner_slice]-params_dict['theta'], window=3)[-1])

plt.figure('Entrainment?')
plt.plot(time.array, fronts, 'b.', label='front')
plt.plot(time.array, c*time.array, 'b-', label='expected front')
plt.plot(time.array, stim_speed*time.array + stim_start, 'r-', label='stim')
plt.legend()
plt.title(f'Sim magintude={stim_magnitude}\nStim speed={stim_speed}')
plt.tight_layout()
plt.savefig(os.path.join(experiment_defaults.media_path, FILE_NAME + '.png'))
