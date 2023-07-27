import numpy as np

from single_step_modifier import single_step_modifier
from space_domain import SpaceDomain
from time_integrator import Euler

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

def generate_stimulus(*, t_on, t_off, c, delta_c, stim_start, stim_width, stim_mag):
    period = t_on + t_off
    stim_speed = c + delta_c
    def stim(x, t):
        x_mapped = x - np.floor(t/period)*period*stim_speed
        t_mapped = t - np.floor(t/period)*period
        stim_vec = np.zeros((2, *x.shape))
        stim_vec[0] = (np.heaviside(stim_start - x_mapped, 0.5) * 
                       np.heaviside(x_mapped - (stim_start - stim_width), 0.5) * 
                       np.heaviside(t_on - t_mapped, 1) * stim_mag)
        return stim_vec
    return stim

def generate_period_time(*, t_on, t_off, c, delta_c, stim_start, stim_width, stim_mag):
    period = t_on + t_off
    def period_time(t):
        return t - np.floor(t/period)*period
    return period_time

def generate_stim_front(*, t_on, t_off, c, delta_c, stim_start, stim_width, stim_mag):
    period = t_on + t_off
    stim_speed = c + delta_c
    def stim_front(t):
        return stim_start + np.floor(t/period)*period*stim_speed
    return stim_front
