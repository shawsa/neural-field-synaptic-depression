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

from neural_field import NeuralField, ParametersBeta, heaviside_firing_rate, exponential_weight_kernel
from space_domain import SpaceDomain

from functools import partial
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix as csr

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

space = SpaceDomain(xis[0], xis[-1], len(xis))

model = NeuralField(
                space=space,
                firing_rate=partial(heaviside_firing_rate,
                                    theta=params_dict['theta']),
                weight_kernel=exponential_weight_kernel,
                params=params)


plt.plot(xis, Us, 'b-')
plt.plot(xis, Qs, 'b--')

N = len(xis)
h = (xis[-1] - xis[0])/(N-1)
data = [-.5]*(N-2) + [.5]*(N-2)
row_index = list(i for i in range(N-2)) + list(i for i in range(N-2))
col_index = list(i for i in range(N-2)) + list(i+2 for i in range(N-2))
D = 1/h*csr((data, (row_index, col_index)))

def fixed_point_iteration(U, Q, mu, alpha, beta, c, **_):
    U_new = c*mu * (D@U) + model.conv(Q*model.firing_rate(U))[1:-1]
    Q_new = -1 + c*alpha * (D@Q) + (beta*Q*model.firing_rate(U))[1:-1]
    return U_new, Q_new

model.conv(xis)

U_iters = [np.heaviside(1 - np.abs(xis), 0)]
Q_iters = [np.ones_like(xis)]

for _ in range(1):
    U_iter = np.zeros_like(xis)
    Q_iter = np.ones_like(xis)
    U_new, Q_new = fixed_point_iteration(U_iters[-1], Q_iters[-1], **params_dict)
    U_iter[1:-1] = U_new
    Q_iter[1:-1] = Q_new
    U_iters.append(U_iter)
    Q_iters.append(Q_iter)

for U in U_iters:
    plt.plot(xis, U)






