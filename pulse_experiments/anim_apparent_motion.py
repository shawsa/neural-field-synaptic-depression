import experiment_defaults

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os.path

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

from root_finding_helpers import find_roots

from time_domain import TimeRay, TimeDomain_Start_Stop_MaxSpacing

from apparent_motion_utils import (
    ShiftingDomain,
    ShiftingEuler,
    ApparentMotionStimulus,
)

FILE_NAME = os.path.join(experiment_defaults.media_path, "apparent_motion.gif")
SAVE_ANIMATION = False

params = ParametersBeta(
    **{
        "alpha": 20.0,
        "beta": 5.0,
        "mu": 1.0,
    }
)
theta = 0.2

USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0509375967740198, 9.553535461425781
    print(f"c={c}\nDelta={Delta}")
else:
    Delta_interval = (7, 20)
    speed_interval = (1, 10)
    Delta = find_delta(
        *Delta_interval, *speed_interval, xs_left, xs_right, verbose=True, **params
    )
    c = find_c(*speed_interval, xs_right, Delta=Delta, verbose=True, **params)

symbol_params = symbolic_dictionary(c=c, Delta=Delta, theta=theta, **params.dict)
U, Q, *_ = get_traveling_pulse(symbol_params, validate=False)


space = ShiftingDomain(-20, 10, 3_001)
model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

solver = ShiftingEuler(shift_tol=1e-4, shift_fraction=4 / 5, space=space)

u0 = np.empty((2, space.num_points))
u0[0] = U(space.array)
u0[1] = Q(space.array)

stim = ApparentMotionStimulus(
    **{
        "t_on": 0.5,
        "t_off": 0.5,
        "speed": c + 0.25,
        # 'speed': c + 0.3,
        "mag": 0.04,
        # 'mag': 0.08,
        "width": 5,
        "start": -0.05,
    }
)

max_time_step = 1e-2
time_step = stim.period / np.ceil(stim.period / max_time_step)
time = TimeRay(0, time_step)
# time = TimeDomain_Start_Stop_MaxSpacing(0, 80, time_step)


def rhs(t, u):
    return model.rhs(t, u) + stim(space.array, t)


try:
    plt.close()
except:
    pass

fig = plt.figure(figsize=(10, 15))
gs = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax_lag_series = fig.add_subplot(gs[0, 1])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])

(theta_line,) = ax0.plot([space.left, space.right], [theta] * 2, "k:")
(stim_line,) = ax0.plot(space.array, stim(space.array, 0)[0], "m-")
(u_line,) = ax0.plot(space.array, u0[0], "b-")
(q_line,) = ax0.plot(space.array, u0[1], "b--")

num_ticks = 6


def plot_callback():
    ax0.set_xlim(space.left, space.right)
    ticks = space.array[:: space.num_points // num_ticks]
    ax0.set_xticks(ticks, [f"{tick:.2f}" for tick in ticks])
    u_line.set_xdata(space.array)
    q_line.set_xdata(space.array)
    stim_line.set_xdata(space.array)
    theta_line.set_xdata([space.left, space.right])


space.callback = plot_callback
space.reset()
times, fronts = map(
    list, zip(*[(i * time.spacing, i * time.spacing * c) for i in range(-199, 1, 1)])
)

time_width = time.spacing * len(times)
(front_natural_line,) = ax1.plot([0, 0 - time_width], [0, 0 - c * time_width], "g-")
(front_stim_line,) = ax1.plot(
    [0, 0 - time_width], [0, 0 - stim.speed * time_width], "m-"
)
(front_line,) = ax1.plot(times, fronts, "b.")
front_window_width = abs(fronts[-1] - fronts[0])
front_window_height = abs(times[-1] - times[0])
ax1.set_xlim(times[-1] - front_window_height, times[-1])
ax1.set_ylim(fronts[-1] - front_window_width, fronts[-1])

relative_front_sample_len = int((stim.t_on + stim.t_off) / time.spacing * 5)
relative_fronts = [0] * relative_front_sample_len
relative_times = [
    stim.period_time(i * time.spacing)
    for i in range(1 - relative_front_sample_len, 1, 1)
]
ax2.plot([0, stim.period], [-stim.width] * 2, "r-")
ax2.plot([stim.t_on] * 2, [-1000, 1000], "g:")
(relative_line,) = ax2.plot(relative_times, relative_fronts, "k.")
(relative_lead_line,) = ax2.plot(0, 0, "go")
ax2.set_xlim(0, stim.t_on + stim.t_off)
ax2.set_ylim(-0.5, 0.5)

ax0.set_xlabel("$x$")
ax1.set_xlabel("$t$")
ax1.set_ylabel("$x$")

ax2.set_xlabel("$t$ (relative to period)")
ax2.set_ylabel("Front lag")

ax_lag_series.set_xlabel("Period")
ax_lag_series.set_ylabel(r"Lag at $T_{off}$")

plt.tight_layout()

sample_freq = 211


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def append_data(self, *args):
        pass


next_period_time = stim.next_off(0)
last_lag = float("inf")
period_index = 0
stop_tol = 1e-5

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_NAME, mode="I")
else:
    writer = NullContext()

with writer:
    for index, (t, (u, q)) in enumerate(
        zip(time, solver.solution_generator(u0, rhs, time))
    ):
        front = find_roots(space.array, u - theta, window=3)[-1]
        rolling_index = index % len(fronts)
        fronts[rolling_index] = front
        times[rolling_index] = t
        relative_rolling_index = index % relative_front_sample_len
        lag = front - stim.front(t)
        relative_fronts[relative_rolling_index] = lag
        relative_times[relative_rolling_index] = stim.period_time(t)
        if abs(t - next_period_time) < time.spacing / 2:
            print(f"{lag=}, \t change={abs(lag-last_lag)}")
            next_period_time = stim.next_off(t + time.spacing)
            if abs(last_lag - lag) < stop_tol:
                print("Entrainment success.")
                break
            elif -lag > stim.width * 1.05:
                print("Entrainment failure.")
                break
            last_lag = lag
            ax_lag_series.plot(period_index, lag, "k.")
            period_index += 1
        if index % sample_freq != 0:
            continue
        u_line.set_ydata(u)
        q_line.set_ydata(q)
        stim_line.set_ydata(stim(space.array, t)[0])
        front_line.set_data(times, fronts)
        front_natural_line.set_data(
            [t, t - time_width], [front, front - c * time_width]
        )
        front_stim_line.set_data(
            [t, t - time_width], [front, front - stim.speed * time_width]
        )
        ax1.set_xlim(times[rolling_index] - front_window_height, times[rolling_index])
        ax1.set_ylim(fronts[rolling_index] - front_window_width, fronts[rolling_index])
        relative_line.set_data(relative_times, relative_fronts)
        relative_lead_line.set_data(
            [relative_times[relative_rolling_index]],
            [relative_fronts[relative_rolling_index]],
        )
        ax2.set_ylim(min(relative_fronts), max(relative_fronts))
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        os.remove(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(1e-3)
