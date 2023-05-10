"""Generate and plot the traveling pulse profile and adjoint nullspace
for the bi-exponential weight kernel and the specified parameters.

The algorithmic and numerical details are desricbed in num_assist.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import experiment_defaults
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2
from plotting_helpers import U_style, Q_style

from scipy.signal import fftconvolve


FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'pulse_diagram')

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

linewidth = 5

xis, Us, Qs = pulse_profile(xs_left, xs_right, **params)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
plt.rc('font', size=20)
active = xis[Us > params['theta']]
axes[0].plot(xis, Us, **U_style, color='blue', linewidth=linewidth, label='Activity')
axes[0].plot(xis, Qs, **Q_style, color='blue', linewidth=linewidth, label='Synaptic Efficacy')
# axes[0].plot(active, 0*active, linewidth=linewidth, color='gray', label='Active Region')
axes[0].plot(xis, params['theta'] + 0*xis, 'k:', linewidth=linewidth, label='Threshold')
axes[0].text(1, .7, 'full resources', fontsize=15)
axes[0].text(-9, .4, 'depression', fontsize=15, rotation=60)
axes[0].text(-25, .4, 'replenishment', fontsize=15, rotation=-25)
axes[0].set_xlim(-35, 15)
axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

loc = -5
axes[0].plot(loc, 0, 'go', markersize = 15)
axes[1].plot(active, Qs[Us > params['theta']], 'b--', linewidth=linewidth, label='$qf[u]$')
axes[1].plot(active, weight_kernel(loc-active), 'g-', linewidth=linewidth, label='$w$')

total_stim = fftconvolve(weight_kernel(xis), np.heaviside(Us - params['theta'], 0)*Qs, mode='same') * (xis[1] - xis[0])
axes[2].plot(xis, total_stim, color='gray', linewidth=linewidth, label='$w\\ast (qf[u])$')

axes[2].set_xlabel('$x$', fontsize=20)
for ax in axes:
    ax.fill_between(active, -1+0*active, 2+0*active, color='gray', alpha=.2, label='Active Region', zorder=-10)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(-.05, 1.05)
plt.tight_layout()
for extension in ['.png']:
    plt.savefig(FILE_NAME + extension)
plt.show()
