#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def random_grad():
    angle = np.random.rand() * 2*np.pi
    return (np.cos(angle), np.sin(angle))

def smooth(s):
    return 3 * s**2 - 2 * s**3

class Perlin:
    def __init__(self, size, num_octaves, start_octave=2):
        self.size = size
        self.components = []
        self.cellSizes = []
        for i in range(num_octaves):
            c = 2**(i+start_octave)

            grads = [
                    [random_grad() for i in range(c+1)] 
                    for j in range(c+1)
                    ]
            self.components.append(grads)
            self.cellSizes.append(float(size) / c)

    def at(self, x, y):
        if x >= self.size:
            x = self.size - 1e-6

        if y >= self.size:
            y = self.size - 1e-6

        result = 0

        mul = 1.0

        maximum = 0

        for i, grads in enumerate(self.components):
            cellSize = float(self.cellSizes[i])

            cx = x / cellSize
            cy = y / cellSize

            x0 = int(cx)
            x1 = x0 + 1
            y0 = int(cy)
            y1 = y0 + 1

            sx = smooth(cx - x0)
            sy = smooth(cy - y0)

            n0 = np.dot((cx-x0, cy-y0), grads[x0][y0])
            n1 = np.dot((cx-x1, cy-y0), grads[x1][y0])
            ix0 = n0 + sx*(n1-n0)
            n0 = np.dot((cx-x0, cy-y1), grads[x0][y1])
            n1 = np.dot((cx-x1, cy-y1), grads[x1][y1])
            ix1 = n0 + sx*(n1-n0)
            result += (ix0 + sy*(ix1-ix0)) * mul

            maximum += mul
            mul *= 0.5
        
        return (result / maximum + 1) / 2

    def __getitem__(self, x):
        return self.at(*x)

