"""Generate and plot the traveling pulse profile and adjoint nullspace
for the bi-exponential weight kernel and the specified parameters.

The algorithmic and numerical details are desricbed in num_assist.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import experiment_defaults
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2


NULLSPACE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bi-exponential nullspace (numerical)')

PULSE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bi-exponential pulse (numerical)')

def weight_kernel(x):
    return .5*np.exp(-np.abs(x))

params = {
    'theta': 0.2,
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0,
    'weight_kernel': weight_kernel
}
params['gamma'] = 1/(1+params['beta'])
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

params['c'] = c
params['Delta'] = Delta

custom = {
    'linewidth': 2.0
}

xs, Us, Qs = pulse_profile(xs_left=xs_left, xs_right=xs_right, **params)
plt.figure('Traveling wave.', figsize=(5, 3))
plt.plot(xs, Us, 'b-', label='$U$', **custom)
plt.plot(xs, Qs, 'b--', label='$Q$', **custom)
# plt.plot([-Delta, 0], [params['theta']]*2, 'k.')
plt.plot(xs, params['theta']+0*xs, 'k:', label='$\\theta$', **custom)
plt.plot([-Delta, 0], [0]*2, color='gray', linewidth=5.0, label='Active Region')
plt.xlim(-30, 20)
plt.xlabel('$\\xi$')
plt.legend(loc='upper left')
plt.title('Traveling Pulse Profile')
plt.tight_layout()
for extension in ['.png', '.eps']:
    plt.savefig(PULSE_FILE_NAME + extension)
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
plt.ylim(-2e-3, 1e-2)
plt.title('bi-exponential nullspace (numerical)')
plt.legend()
for extension in ['.png', '.eps']:
    plt.savefig(NULLSPACE_FILE_NAME + extension)
plt.show()
