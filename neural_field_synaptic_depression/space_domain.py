'''
A class for the spatial domain of the neural field.
'''

import numpy as np
from math import ceil

class SpaceDomain:
    def __init__(self, left: float, right: float, num_points: int):
        self.left = left
        self.right = right
        self.num_points = num_points
        self.spacing = (right-left)/(num_points - 1)

        self.array = np.linspace(self.left, self.right, self.num_points)


class BufferedSpaceDomain(SpaceDomain):
    def __init__(self,
                 left: float,
                 right: float,
                 num_points: int,
                 buffer_percent: float):

        num_buffer = ceil(num_points * buffer_percent)
        spacing = (right - left)/(num_points - 1)
        buff_left = left - num_buffer*spacing
        buff_right = right + num_buffer*spacing

        super().__init__(buff_left, buff_right, num_points + 2*num_buffer)

        self.inner_slice = slice(num_buffer, -num_buffer)

    @property
    def inner(self):
        return self.array[self.inner_slice]
