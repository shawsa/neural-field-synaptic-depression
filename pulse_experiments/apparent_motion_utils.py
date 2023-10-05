import numpy as np

from neural_field_synaptic_depression.single_step_modifier import single_step_modifier
from neural_field_synaptic_depression.space_domain import SpaceDomain
from neural_field_synaptic_depression.time_integrator import Euler

class ShiftingDomain(SpaceDomain):

    def __init__(self, left: float, right: float, num_points: int):
        self.original_values = left, right, num_points
        self.left = left
        self.right = right
        self.num_points = num_points
        self.spacing = (right-left)/(num_points - 1)
        
        self.callback = None

    @property
    def array(self):
        return np.linspace(self.left, self.right, self.num_points)

    def shift(self, int_shift: int):
        old_arr = self.array
        self.left = self.array[-int_shift]
        self.right = self.left + (old_arr[-1] - old_arr[0])
        if self.callback is not None:
            self.callback()

    def reset(self):
        self.left, self.right, self.num_points = self.original_values
        if self.callback is not None:
            self.callback()

def get_shift_callback(shift_tol: float, shift_fraction: float, space: ShiftingDomain):
    def shift_predicate(t, u, f, h):
        return u[0][-1] > shift_tol
    int_shift = int(shift_fraction*space.num_points)

    def shift_solution_callback(t, u, f, h):
        if not shift_predicate(t, u, f, h):
            return t, u, f, h
        u_new = np.empty_like(u)
        u_new[0] = 0
        u_new[1] = 1
        u_new[0, :int_shift] = u[0, -int_shift:]
        u_new[1, :int_shift] = u[1, -int_shift:]
        space.shift(int_shift)
        return t, u_new, f, h
    return shift_solution_callback

def ShiftingEuler(shift_tol: float, shift_fraction: float, space: ShiftingDomain):
    return single_step_modifier(get_shift_callback(shift_tol, shift_fraction, space))(Euler)()

class ApparentMotionStimulus:
    def __init__(self, *, t_on, t_off, speed, start, width, mag):
        self.t_on = t_on
        self.t_off = t_off
        self.speed = speed
        self.start = start
        self.width = width
        self.mag = mag

        self.period = t_on + t_off

    def __call__(self, x, t):
        x_mapped = x - np.floor(t/self.period)*self.period*self.speed
        t_mapped = t - np.floor(t/self.period)*self.period
        stim_vec = np.zeros((2, *x.shape))
        stim_vec[0] = (np.heaviside(self.start - x_mapped, 0.5) * 
                       np.heaviside(x_mapped - (self.start - self.width), 0.5) * 
                       np.heaviside(self.t_on - t_mapped, 1) * self.mag)
        return stim_vec

    def period_time(self, t):
        return t - np.floor(t/self.period)*self.period

    def front(self, t):
        return self.start + np.floor(t/self.period)*self.period*self.speed

    def next_on(self, t):
        return (1 + np.floor(t/self.period))*self.period

    def next_off(self, t):
        period_start = np.floor(t/self.period)*self.period
        if t - period_start < self.t_on:
            # stimulus is currently on
            return period_start + self.t_on
        # stimulus is off
        return period_start + self.period + self.t_on
