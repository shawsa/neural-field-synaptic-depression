'''
A class for the spatial domain of the neural field.
'''

import numpy as np

class SpaceDomain:
    def __init__(self, left: float, right: float, num_points: int):
        self.left = left
        self.right = right
        self.num_points = num_points
        self.spacing = (right-left)/(num_points - 1)

        self.array = np.linspace(self.left, self.right, self.num_points)
