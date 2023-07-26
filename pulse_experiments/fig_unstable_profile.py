"""Generate and plot the traveling pulse profile and adjoint nullspace
for the bi-exponential weight kernel and the specified parameters.

The algorithmic and numerical details are desricbed in num_assist.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import experiment_defaults

from helper_symbolics import (
        find_symbol_by_string,
        free_symbols_in,
        expr_dict,
        get_traveling_pulse)

PULSE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'unstable_profile')

params = {
    'theta': 0.2,
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0,
}

# using old beta
params['beta'] /= params['alpha']


c, Delta = 0.2282045781557908, 1.7456243859964509

params['c'] = c
params['Delta'] = Delta

# using an ancient sympy incantation
symbol_set = free_symbols_in([
        recursive_reduce(expr)
        for expr in expr_dict['speed_width_conditions']
])

symbol_params = {find_symbol_by_string(symbol_set, key): value
                 for key, value in params.items()}

U, Q, *_ = get_traveling_pulse(symbol_params, validate=False)

custom = {
    'linewidth': 2.0
}

xs = np.linspace(-200, 200, 8001)
plt.figure('Traveling wave.', figsize=(5, 3))
plt.plot(xs, U(xs), 'b-', label='$U$', **custom)
plt.plot(xs, Q(xs), 'b--', label='$Q$', **custom)
# plt.plot([-Delta, 0], [params['theta']]*2, 'k.')
plt.plot(xs, params['theta']+0*xs, 'k:', label='$\\theta$', **custom)
plt.plot([-Delta, 0], [0]*2, color='gray', linewidth=5.0, label='Active Region')
plt.xlim(-30, 20)
plt.ylim(-.1, 1.1)
plt.xlabel('$\\xi$')
plt.legend(loc='upper left')
plt.title('Unstable Pulse Profile')
plt.tight_layout()
for extension in ['.png', '.eps', '.pdf']:
    plt.savefig(PULSE_FILE_NAME + extension)
plt.show()
