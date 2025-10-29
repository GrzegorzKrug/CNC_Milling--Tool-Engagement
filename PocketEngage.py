
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.style import use

from yasiu_math.convolve import moving_average
from yasiu_native.time import measure_real_time_decorator

# from numba import jit, njit
import math
import matplotlib.pyplot as plt


def CalcEngageEntryParallel(e):
    r = 1
    D = 2
    e = np.clip(e, 0, 50)
    # x1 = D * (50 - e) / 100
    x1 = r - e * D / 100
    x2 = np.sqrt(r - x1 * x1)
    E = x2 * 100
    return E


def CalcEngageEntryDiagonal(e, alfa=45):
    r = 1
    d = 2
    x1 = 1 - 2 * e / 100

    beta = np.acos(x1)
    # beta=beta*0

    alfaRad = np.deg2rad(alfa)
    angleSum = beta + alfaRad
    angleSum = np.clip(angleSum, 0, np.pi / 2)
    E = np.sin(angleSum) * 100

    return E


if __name__ == "__main__":
    use('ggplot')
    engage = np.linspace(0, 80, 1000)

    ENG1 = CalcEngageEntryParallel(engage)
    ENG2 = CalcEngageEntryDiagonal(engage, alfa=45)

    plt.figure(figsize=(9, 7), dpi=150)
    plt.plot(engage, ENG1, label="Engage Parrarel")
    plt.plot(engage, ENG2, label=f"Engage Diagonal: {45}°")

    plt.grid(True)
    plt.title("Engage")
    plt.legend()
    plt.xlabel("Path engage %")
    plt.ylabel("Engage when cutting new layer %")
    ax = plt.gca()
    tcks = ax.get_yticks()
    tcks = np.arange(0, 100.001, 10, dtype=float)
    tcks[7] = 70.71
    # tcks = np.concat([tcks, [70.71]])
    ax.set_yticks(tcks)

    plt.tight_layout()
    name = os.path.join("images", "PocketEngage.png")
    plt.savefig(name)
    plt.show()
