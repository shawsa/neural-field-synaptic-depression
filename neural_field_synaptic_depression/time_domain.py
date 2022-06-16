'''A time-discretization for an ODE solver.'''


import math
import numpy as np

class TimeDomain:
    
    def __init__(self, start, spacing, steps):
        self.start = start
        self.spacing = spacing
        self.steps = steps
        self.initialze_array()

    def initialze_array(self):
        self.array = self.start + self.spacing*np.arange(self.steps+1)

class TimeDomain_Start_Stop_MaxSpacing(TimeDomain):

    def __init__(self, start, stop, max_spacing):
        self.start = start
        self.steps = math.ceil((stop-start)/max_spacing)
        self.spacing = (stop - start)/self.steps
        self.initialze_array()


class TimeDomain_Start_Stop_Steps(TimeDomain):
    def __init__(self, start, stop, steps):
        self.start = start
        self.steps = steps
        self.spacing = (stop - start)/steps
        self.initialze_array()

