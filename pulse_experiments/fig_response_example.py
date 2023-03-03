"""Generate a figure similar to Fig 2. in Kilpatrik Ermentrout 2012.
Use a delta for the stimulus variable, and use a bi-exponential
traveling pulse.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import experiment_defaults
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2, local_interp

from functools import partial
from neural_field import NeuralField, ParametersBeta, heaviside_firing_rate
from plotting_helpers import make_animation
from space_domain import SpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper


FIG_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'response_example.eps')


def weight_kernel(x):
    return .5*np.exp(-np.abs(x))

params = ParametersBeta(**{
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0,
})
params_dict = {**params.dict,
        'gamma': params.gamma,
        'theta': 0.2,
        'weight_kernel': weight_kernel
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

space = SpaceDomain(-100, 200, 10**3)
time = TimeDomain_Start_Stop_MaxSpacing(0, 60, 1e-2)

initial_offset = 0
u0 = np.empty((2, space.num_points))
u0[0] = np.array([local_interp(x, xis, Us) for x in space.array])
u0[1] = np.array([local_interp(x, xis, Qs) for x in space.array])

model = NeuralField(
                space=space,
                firing_rate=partial(heaviside_firing_rate,
                                    theta=params_dict['theta']),
                weight_kernel=weight_kernel,
                params=params)

delta_time = 20
epsilon = 0.1
stim_profile = np.ones_like(u0)

solver = TqdmWrapper(EulerDelta(delta_time, epsilon*stim_profile))

print('solving purturbed case')
us = solver.solve(u0, model.rhs, time)

# unperturbed
print('solving unpurturbed case')
unperturbed_solver = TqdmWrapper(Euler())
us_unperturbed = unperturbed_solver.solve(u0, model.rhs, time)


plt.figure('Wave Response Example', figsize=(7, 3))
for index, color in [(0, 'b'),
                     (int(100*delta_time)+1, 'm'),
                     (int(100*delta_time)+200, 'c'),
                     (-1, 'g')]:
    plt.plot(space.array, us_unperturbed[index][0], linewidth=4.0, color=color, linestyle='--')
    plt.plot(space.array, us[index][0], linewidth=4.0, color=color)
    plt.plot([], [], color=color, label=f'$t={time.array[index]:.0f}$')
plt.xlim(-20, 70)
plt.plot([], [], 'k-', label='perturbed')
plt.plot([], [], 'k--', label='unperturbed')
plt.plot(space.array, params_dict['theta']+0*space.array, 'k:', label='$\\theta$')
plt.ylabel('$u$')
plt.xlabel('$x$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(FIG_FILE_NAME)
