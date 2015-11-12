# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

B = np.array([[1,1,1,1,1,1], [2,2,2,2,2,2]]) # matrix D x N, where D is the number of dimensions, N is the number of samples
print B
# plt.plot(np.linspace(1,10), np.linspace(1,10))
# plt.show()

n = 60 # thus, N = n^2. As I'm sampling from a plane, D = 2
grid_points = np.arange(n)
xx, yy = np.meshgrid(grid_points)
print [(x,y) for x in xx.flatten() for y in yy.flatten()]
