import os
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from yasiu_math.convolve import moving_average
from yasiu_native.time import measure_real_time_decorator

# from numba import jit, njit
import math
import matplotlib.pyplot as plt


rad = np.linspace(2, 6, 100)
chipload = np.linspace(0.1, 0.6, 100)
XRadius, YChip = np.meshgrid(rad, chipload)
w = rad - np.sqrt(np.pow(XRadius, 2) - np.pow(YChip / 2, 2))

plt.figure(figsize=(8, 5), dpi=150)
plt.contourf(XRadius, YChip, w, levels=15, cmap='turbo_r')
plt.colorbar()

contours = plt.contour(XRadius, YChip, w, levels=15, colors='black')
plt.clabel(contours, inline=True, fontsize=10)

plt.title("Left over material width")
plt.ylabel("Chipload [mm]")
plt.xlabel("Radius [mm]")
plt.grid()
plt.tight_layout()
plt.show()
