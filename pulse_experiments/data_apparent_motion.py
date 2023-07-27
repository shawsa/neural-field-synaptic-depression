
import experiment_defaults
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from itertools import islice

from neural_field import (
        NeuralField,
        ParametersBeta,
        heaviside_firing_rate,
        exponential_weight_kernel,
)

from helper_symbolics import (
        expr_dict,
        find_symbol_by_string,
        free_symbols_in,
        get_traveling_pulse,
        recursive_reduce,
        symbolic_dictionary,
)

from time_domain import TimeRay

from apparent_motion_utils import (
        ShiftingDomain,
        ShiftingEuler,
        generate_stimulus,
)

default = ParametersBeta(**{
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0,
    'theta': 0.2
})

c, Delta = 1.0509375967740198, 9.553535461425781
Delta_interval = (7, 20)
speed_interval = (1, 10)
Delta = find_delta(*Delta_interval, *speed_interval,
                   xs_left, xs_right, verbose=True, **params)
c = find_c(*speed_interval,  xs_right,
           Delta=Delta, verbose=True, **params)

symbol_params = symbolic_dictionary(c=c, Delta=Delta, theta=theta, **params.dict)
U, Q, *_ = get_traveling_pulse(symbol_params, validate=False)


space = ShiftingDomain(-20, 40, 6_001)
time = TimeRay(0, 1e-2)

model = NeuralField(
            space=space,
            firing_rate=partial(heaviside_firing_rate, theta=theta),
            weight_kernel=exponential_weight_kernel,
            params=params)

solver = ShiftingEuler(shift_tol=1e-6, shift_fraction=2/3, space=space)

u0 = np.empty((2, space.num_points))
u0[0] = U(space.array)
u0[1] = Q(space.array)

# Stimulus
stim = generate_stimulus(
        t_on = .5,
        t_off = 0.5,
        delta_c = 0.25,
        stim_mag = 0.04,
        stim_width = 5,
        stim_start = -.05,
        c = c
)

def rhs(t, u):
    return model.rhs(t, u) + stim(space.array, t)
          

try:
    plt.close()
except:
    pass

plt.figure()
theta_line, = plt.plot([space.left, space.right], [theta]*2, 'k:')
stim_line, = plt.plot(space.array, stim(space.array, 0)[0], 'm-')
u_line, = plt.plot(space.array, u0[0], 'b-')
q_line, = plt.plot(space.array, u0[1], 'b--')

num_ticks = 6
def plot_callback():
    plt.xlim(space.left, space.right)
    ticks = space.array[::space.num_points//num_ticks]
    plt.xticks(ticks, [f'{tick:.2f}' for tick in ticks])
    u_line.set_xdata(space.array)
    q_line.set_xdata(space.array)
    stim_line.set_xdata(space.array)
    theta_line.set_xdata([space.left, space.right])

space.callback = plot_callback
space.reset()
sample_freq = 251
for t, (u, q) in islice(zip(time,
                            solver.solution_generator(u0, rhs, time)),
                        0, None, sample_freq):
    u_line.set_ydata(u)
    q_line.set_ydata(q)
    stim_line.set_ydata(stim(space.array, t)[0])
    plt.pause(1e-3)
    if np.max(u) < theta:
        break
