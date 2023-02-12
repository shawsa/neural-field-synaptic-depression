"""Generate and plot the traveling pulse profile and adjoint nullspace
for the wizzard hat  weight kernel (a laterally inhibitory kernel)
and the specified parameters.

The algorithmic and numerical details are desricbed in num_assist.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import experiment_defaults
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2


NULLSPACE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'wizzard-hat nullspace (numerical).png')

PULSE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'wizzard-hat pulse (numerical).png')


def weight_kernel(x):
    abs_x = np.abs(x)
    return (1-.5*abs_x)*np.exp(-abs_x)


params = {
    'theta': 0.2,
    'alpha': 20,
    'beta': 5.0,
    'mu': 1.0,
    'weight_kernel': weight_kernel
}
params['gamma'] = 1/(1+params['beta'])
xs_right = Domain(0, 200, 8001)
xs_left = Domain(-200, 0, 8001)

USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 0.6157679207346518, 5.515222549438477
    print(f'c={c}\nDelta={Delta}')
else:
    Delta_interval = (5, 10)
    speed_interval = (.1, 4)
    Delta = find_delta(*Delta_interval, *speed_interval,
                       xs_left, xs_right, verbose=True, **params)
    c = find_c(*speed_interval,  xs_right,
               Delta=Delta, verbose=True, **params)

params['c'] = c
params['Delta'] = Delta

xs, Us, Qs = pulse_profile(xs_right, xs_left, **params)
plt.figure('Traveling wave.')
plt.plot(xs, Us, 'b-', label='$U$')
plt.plot(xs, Qs, 'b--', label='$Q$')
plt.plot([-Delta, 0], [params['theta']]*2, 'k.')
plt.xlim(-30, 20)
plt.legend()
plt.title('Traveling Pulse (numerical)')
plt.savefig(PULSE_FILE_NAME)
plt.show()

A0, AmD = nullspace_amplitudes(xs, Us, Qs, **params)
print(f'A_0={A0}\tA_{{-Delta}}={AmD}')
params['A0'] = A0
params['AmD'] = AmD
v1_arr = v1(xs, **params)
zs = Domain(-Delta, 0, 201)
v2_arr = v2(xs, **params)

plt.figure('Nullspace')
plt.plot(xs, v1_arr, label='$v_1$')
plt.plot(xs, v2_arr, label='$v_2$')
plt.xlim(-15, 15)
plt.ylim(-4e-3, 1e-2)
plt.title('bi-exponential nullspace (numerical)')
plt.legend()
plt.savefig(NULLSPACE_FILE_NAME)
plt.show()
