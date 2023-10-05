"""A driver demonstrating how to use the package to run neural field simulations.
This driver will us a periodic stimulus to generate a series of counterpropagating
pairs of traveling pulses.
"""

import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from itertools import islice
from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.time_domain import (
        TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import (
        Euler,
)
from neural_field_synaptic_depression.time_integrator_tqdm import (
        TqdmWrapper,
)
from neural_field_synaptic_depression.space_domain import (
        SpaceDomain,
)


# these parameter values are chosen so that we are in the pulse regime.
params = ParametersBeta(
    **{
        "alpha": 20.0,
        "beta": 5.0,
        "mu": 1.0,
    }
)
theta = 0.2

# define the space and time domains and discretizations
space = SpaceDomain(left=-500, right=500, num_points=5_000)
time = TimeDomain_Start_Stop_MaxSpacing(start=0, stop=1_000, max_spacing=1e-2)

# create a model object used to generate a RHS forcing function
# the model uses an FFT to quickly compute the convolution.
model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

# initial conditions
# activity is zero
initial = np.zeros((2, len(space.array)))
# synaptic efficacy is 1
initial[1] = 1 + 0*space.array

# create an ODE solver object to integrate the solution in time
# we will wrap the solver in a TqdmWrapper object that will
# show progress bars as the solution is being calculated.
solver = TqdmWrapper(Euler())


stim_period = 300  # amount of time between stims
stim_on = 2  # duration of eac stim
def stim(t):
    """An autonomous stimulus function (I_u in the paper)."""
    ret = np.zeros_like(initial)
    stim_factor = 0
    if t - np.floor(t/stim_period)*stim_period < stim_on:
        stim_factor = 1

    ret[0] = 0.3 * stim_factor * np.exp(-space.array**2)
    return ret


def rhs(t, u):
    """The right-hand-side forcing function for our model."""
    return model.rhs(t, u) + stim(t)


plt.ion()
plt.figure(figsize=(10, 5))
u_line, = plt.plot(space.array, 0*space.array, "b-", label="$u$")
q_line, = plt.plot(space.array, 1 + 0*space.array, "b--", label="$q$")
stim_line, = plt.plot(space.array, stim(0)[0], "m-", label="$I_u$")
plt.xlabel("$x$")
plt.xlim(space.array[0], space.array[-1])
plt.ylim(-.1, 1.2)
time_text = plt.text(0, 1.1, f"$t={0}$")
plt.legend(loc="upper right")

# we use isclice to only plot every 10th frame
steps_per_frame = 50
for t, (u, q) in islice(
        zip(time.array,
            solver.solution_generator(u0=initial, rhs=rhs, time=time)),
        0,  # start
        None,  # stop
        steps_per_frame):  # step
    u_line.set_ydata(u)
    q_line.set_ydata(q)
    stim_line.set_ydata(stim(t)[0])
    time_text.set_text(f"$t={t}$")
    plt.pause(1e-3)
