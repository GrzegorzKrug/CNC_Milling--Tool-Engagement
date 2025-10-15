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


from CirclesOffset import findCutDistance


def estimation(xdeg, c, r):
    xrad = np.deg2rad(xdeg)
    up = np.atan(c / r)
    # up = c/r
    return xdeg * 0 + up
    # return xdeg


if __name__ == "__main__":
    pass
    XDeg = np.linspace(-90, -89, 200)
    YRes = XDeg * 0
    # ChipRev=4
    Radius = 6
    Chip = 5
    combinations = [
        Chip / Radius,
        Radius / Chip,
        2 * Chip / Radius,
        2 * Radius / Chip,
        Chip / Radius / 2,
        Radius / Chip / 2,
    ]
    combinations = [*combinations, *[-c for c in combinations]]

    funs = [
        np.sin, np.cos, np.tan, np.atan, np.asin, np.acos
    ]
    Res = np.zeros((len(funs), len(combinations)), dtype=np.double)
    for fi, f in enumerate(funs):
        print(f.__name__)
        for ci, c in enumerate(combinations):
            # print(f"{f(c):>7.3f} ", end='')
            d = f(c)
            Res[fi, ci] = d
        # print()
    print(Res.round(4))
    # vals = Res.ravel().reshape(1, -1)
    # plt.plot((vals + vals.T).ravel())
    # plt.show()

    for wi, w in enumerate(XDeg):
        dist = findCutDistance(np.deg2rad(w), Radius, Chip)
        YRes[wi] = dist

    # plt.close("all")
    # plt.plot(XDeg, np.cos(np.deg2rad(XDeg)) * Chip, color='gray', alpha=0.5)
    plt.figure(figsize=(12, 8))
    plt.plot(XDeg, YRes, label="True model")
    plt.plot(XDeg, estimation(XDeg, Chip, Radius), label="my custom function")
    plt.grid()
    # plt.xticks(np.arange(-90, 90.01, 15))
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
