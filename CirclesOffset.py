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


def findPoint(R=10, C=2, alpha_deg=30):
    alfaRadians = alpha_deg * np.pi / 180
    m = np.tan(alfaRadians)
    p1 = np.array((np.cos(alfaRadians) * R, np.sin(alfaRadians) * R + C))

    # Solve quadratic for intersection with Circle 1
    A = 1 + m**2
    B = 2 * m * C
    D = B**2 - 4 * A * (C**2 - R**2)

    if D < 0:
        print("No real intersection.")
        return -1

    sqrtD = np.sqrt(D)
    x1 = (-B + sqrtD) / (2 * A)
    x2 = (-B - sqrtD) / (2 * A)
    y1 = m * x1 + C
    y2 = m * x2 + C

    # Choose intersection in same direction as point on Circle 2
    P2x, P2y = R * math.cos(alfaRadians), C + R * math.sin(alfaRadians)
    chosen = (x1, y1) if x1 * P2x > 0 else (x2, y2)

    return np.sqrt((np.pow(chosen - p1, 2)).sum())


def drawCircles(R, C):
    y_int = C / 2
    x_int = math.sqrt(R**2 - (C**2) / 4)
    points = [(-x_int, y_int), (x_int, y_int)]

    # Generate circle coordinates
    theta = [math.radians(t) for t in range(361)]
    x_circle = [R * math.cos(t) for t in theta]
    y_circle1 = [R * math.sin(t) for t in theta]
    y_circle2 = [C + R * math.sin(t) for t in theta]

    # Plot circles
    plt.figure(figsize=(6, 6))
    plt.plot(x_circle, y_circle1, label="Circle 1 (center (0,0))")
    plt.plot(x_circle, y_circle2, label=f"Circle 2 (center (0,{C}))")


def test1Point():
    RAD = 5
    CHIP = 1
    alfa = 70

    dist = findPoint(R=RAD, C=CHIP, alpha_deg=alfa)
    print(f"Dist: {dist}")

    drawCircles(RAD, CHIP)
    P1 = (np.cos(alfa * np.pi / 180) * RAD, np.sin(alfa * np.pi / 180) * RAD + CHIP)
    plt.title(f"Distance: {dist}")
    plt.scatter(*P1)

    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def CompareCirlceCost():
    RAD = 10
    # CHIP = 2
    for CHIP in [2, 4, 5]:
        plt.figure()
        Alfa = np.linspace(0, 90, 100)
        ResCos = np.cos((90 - Alfa) * np.pi / 180) * CHIP
        ResDist = Alfa * 0

        for ai, al in enumerate(Alfa):
            dt = findPoint(R=RAD, C=CHIP, alpha_deg=al)
            ResDist[ai] = dt

        plt.plot(Alfa, ResDist, label="2CircleDist")
        plt.plot(Alfa, ResCos, label="cos")

        print(ResDist.min() / RAD)
        # plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.title(f"MinDist: {ResDist.min():>3.2f}")
        plt.tight_layout()

    plt.show()


def solveIntersection(r, c, a, b, *, wRad=0, XPrec=None):
    if XPrec is None:
        AMOUNT = 2000
        if -np.pi / 4 < wRad < np.pi / 4:
            "Center"
            X = np.linspace(- r / 1.5, r / 1.5, AMOUNT, dtype=np.double)
        elif wRad < 0:
            "Negative"
            X = np.linspace(- r, 0, AMOUNT, dtype=np.double)
        else:
            "Positive"
            X = np.linspace(0, r, AMOUNT, dtype=np.double)
    else:
        X = np.linspace(XPrec - r / 4, XPrec + r / 4, 50000, dtype=np.double)
        # X = np.linspace(XPrec - c / r*2, XPrec + c / r*2, 10000, dtype=np.double)
        X = np.clip(X, -r, r)

    y1 = a * X + b

    # wDeg = X / r * 90
    "Inverse map of X = sin(rad)"
    wRad = np.arcsin(X / r)
    wRad[np.isnan(wRad)] = 0
    # wRad = np.deg2rad(wDeg)

    # x2 = np.sin(wRad) * r
    # wRad = np.arcsin(wRad)
    Line1y = np.cos(wRad) * r + np.sin(wRad) * (c / 2) - c / 2 - c
    # y2 = np.cos(wRad) * r + np.sin(wRad) * ch - ch / 2
    y2 = Line1y
    # plt.scatter(wDeg / 90 * Radius, y2)
    diff = np.abs(y2 - y1).astype(float)
    # diff[np.isnan(diff)] = r
    # plt.plot(wDeg)
    # plt.plot(x, x2, linewidth=3)
    # plt.plot(X, y2, linewidth=3, color='red')
    xind = np.argmin(diff)
    xRes = X[xind]
    yRes = a * xRes + b
    # print(diff)
    # print(f"Min: {diff.min()}")
    # print("Xind:", xind)
    # print(X[[xind, xind]])
    # plt.plot(X, diff, color='green')
    # plt.plot(X[[xind, xind]], [0, 10])
    # plt.plot(X, np.isnan(X), color='black')
    # plt.plot(X)
    # plt.scatter(xRes, yRes, marker='.', s=50)
    return (xRes, yRes)


def findCutDistance(wRad, radius, Chip, *, plotLine=False):
    """
        wRad - angle in radians
        radius - radius of cutter
        Chip - distance to move
    """
    "P0 is moving center for current cut"
    "P1 is previous cut"
    "P2 is current cut"
    P2x = np.sin(wRad) * radius
    P2y = np.cos(wRad) * radius + np.sin(wRad) * (Chip / 2) - Chip / 2
    # plt.scatter(P2x, P2y)

    P0x = wRad * 0
    # P0y = np.sin(wRad) * Chip / 2 - Chip / 2
    P0y = (np.sin(wRad) - 1) * Chip / 2
    # plt.scatter(P0x, P0y)
    # plt.scatter(P2x, P2y)
    # plt.plot([P0x, P2x], [P0y, P2y], color='gray', alpha=0.5)

    "Line Intersecting P2 and P0"
    "y = ax + b"
    # wDeg = np.linspace(5, 25, 20)
    a = (P2y - P0y) / (P2x - P0x)
    b = P0y - a * P0x
    x = np.linspace(0, radius, 50)
    LineI = a * x + b
    mask = LineI <= radius
    x = x[mask]
    LineI = LineI[mask]
    # plt.plot(x, LineI, color='green', alpha=0.3)
    # print(a, b)

    # y = np.sin(wRad) * Radius + np.cos(wRad) * Chip
    # P1y = np.sin(wRad) * Radius + np.cos(wRad) * Chip - Chip
    # P1x = (P1y - b) / a
    P1x, P1y = solveIntersection(radius, Chip, a, b, wRad=wRad)
    P1x, P1y = solveIntersection(radius, Chip, a, b, XPrec=P1x)
    # P1x, P1y = solveIntersection(radius, Chip, a, b, XPrec=P1x)
    if plotLine:
        plt.plot([P0x, P1x], [P0y, P1y], color='gray', alpha=0.2)
        plt.scatter(P1x, P1y, color='lightgreen')
        # CutDeg =
        color = "orange" if np.rad2deg(wRad) > -95.8 else "purple"
        plt.plot([P1x, P2x], [P1y, P2y], color=color, alpha=0.8, linewidth=3)

    dist = np.sqrt(np.pow(P1x - P2x, 2) + np.pow(P1y - P2y, 2))
    return dist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ModelFunction(deg, chip, radius):
    out = np.cos(np.deg2rad(deg))
    up = np.atan(chip / radius) / chip  # UP factor
    center = np.tan(chip / radius)
    center = -np.tan(Radius / Chip) / 16 / 1.5

    "UP 1"
    # up = up * (1 - np.sin(np.deg2rad((deg + 90))))
    # out = out + up

    "UP with cosin"
    # c = np.cos(np.deg2rad((deg + 90) / 2))  # Falling <1, 0> <-90, 90>
    # up = up * c
    # out = out + up
    # y = (chip / (2 * radius)) * (1 - np.cos(np.deg2rad(deg)))
    # return y
    "UP left half cosin"
    # c = np.cos(np.deg2rad((deg + 90) * 2))  # Falling <1, 0> <-90, 90>
    # return c
    # up = up * c
    # out = out + up
    "GPT"
    theta = np.radians(deg)

    # Vertical shift from your ±90° value
    D = np.arctan(chip / radius) / chip  # verify your estimation
    # Verify calculation
    # np.arctan(2/6)/2 ≈ 0.1618 (this is too high; maybe adjust scale)
    D = 0.0324

    # Horizontal shift for peak
    theta0 = np.radians(13.4)

    # Amplitude and frequency
    A = 2.0415 - D           # peak minus baseline
    B = 1.0                  # will tweak to match ±90° value

    # Scale B to match ±90° close to baseline
    # Simple approach: scale so cos(B*(90-θ0)) ≈ (0.0324 - D)/A
    target = (0.0324 - D) / A
    B = np.arccos(target) / np.radians(90 - 13.4)

    # Final function
    return A * np.cos(B * (theta - theta0)) + D
    return out * chip


if __name__ == "__main__":
    ""
    wDeg = np.linspace(-70, 80, 20)
    wRad = np.deg2rad(wDeg)
    Flute = 2
    Radius = 6
    ChipRev = 2
    Chip = ChipRev / Flute

    "For cuts"
    wDeg = np.linspace(0, 180, 50)
    wRad = np.deg2rad(wDeg)

    "P1 Under"
    x = np.cos(wRad) * Radius
    # y = np.sin(wRad) * Radius + np.sin(np.pi / 2 - wRad) * Chip
    y = np.sin(wRad) * Radius + np.cos(wRad) * (Chip / 2) - Chip / 2
    plt.figure(figsize=(10, 7))
    plt.plot(x, y - Chip, color="green", label="Previous cut")
    plt.plot(x, y, label="Current cut", color='blue', linewidth=2)

    # XDeg = np.linspace(-115, -90, 40)  # Rubbing angle is changing

    "For thicknes estimation"
    XDeg = np.linspace(-90, 90, 80)

    EngageY = XDeg * 0
    for wi, w in enumerate(XDeg):
        wRad = np.deg2rad(w)
        dist = findCutDistance(wRad, Radius, Chip, plotLine=True)
        EngageY[wi] = dist
        # print(f"{w:>3.2f}, {dist:>3.5f}")

    XRad = np.deg2rad(XDeg)
    tempY = np.cos(XRad) * Radius
    tempX = np.sin(XRad) * Radius
    plt.plot(
        tempX, tempY, color='black', alpha=0.5, dashes=[5, 3],
        label="Stationary circles (for reference)"
    )
    plt.plot(tempX, tempY - Chip, color='black', alpha=0.5, dashes=[5, 3])
    plt.plot(tempX, tempY - 2 * Chip, color='black', alpha=0.5, dashes=[5, 3])
    plt.plot(
        [0, 0], [-Chip, 0], label="ChipLoad per 1 blade ( 180° )",
        linewidth=3, color='red', alpha=0.8
    )
    print(Radius, Chip)
    RubWidth = Radius - np.sqrt(np.pow(Radius, 2) - np.pow(Chip / 2, 2))
    print("Rub Width:", RubWidth)
    # plt.plot(
    # [-Radius, -Radius + RubWidth], [-Chip * 1.5, -Chip * 1.5],
    # color='Cyan', linewidth=3, label='Rub Width'
    # )

    # plt.plot(XDeg / 90 * Radius, tempY + Chip, label="Cos function", alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLine = Line2D([0, 0], [0, 1], color='orange', alpha=1, linewidth=3)
    handles.append(newLine)
    labels.append("Wood thickness")

    plt.grid(True)
    plt.legend(handles=handles, labels=labels, loc='lower right')
    plt.title(f"Cutting comparison, Radius: {Radius}, Chipload: {Chip:<2.2f}")
    plt.xlabel("Y Distance")
    plt.ylabel("X Distance")
    plt.tight_layout()
    # plt.show()

    "Difference plot"
    # plt.figure()
    # diff = EngageY - np.cos(XRad) * Chip
    # plt.plot(XDeg, diff, label="Amplified value in real model")
    # plt.title("Difference between 'Cos' model and real model.")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()

    # plt.close("all")
    plt.figure(figsize=(12, 8))
    # plt.subplot(1, 2, 1)
    # Y = moving_average(Y, 5, "keep")
    plt.plot(XDeg, EngageY, label="Moving cutter model", color='green', linewidth=3)

    XRad = np.deg2rad(XDeg)
    tempY = np.cos(XRad) * Chip  # + np.sin(XRad) * Chip
    plt.plot(XDeg, tempY, color='blue', label="Cos function", alpha=0.5)

    up = 0.05
    center = 0.05
    up = np.tan(Chip / Radius) / 2
    center = -np.tan(Radius / Chip) / 16 / 1.5
    # shifted = cosShifted(XDeg / 90, up, center) * Chip
    shifted = ModelFunction(XDeg, Chip, Radius)
    # plt.plot(XDeg, shifted, label="Shifted", color='red')
    diff = tempY - EngageY
    plt.grid(True)
    # aprox = np.cos(XRad) + sigmoid(X / 90) / 10
    # plt.plot(XDeg, shifted, label="Custom shifted", color='red')
    # plt.xticks(np.arange(-90, 90.01, 30))

    # plt.subplot(1, 2, 2)
    # plt.plot(XDeg, np.sin(XRad), label='sin')
    # plt.plot(XDeg, np.cos(XRad), label='cos')
    # plt.plot(XDeg, np.atan(XRad), label='tan')
    # plt.figure()
    # plt.plot(X, diff, label="Cos function")

    plt.legend()
    # plt.scatter(8.91, 4.32)
    # plt.scatter(P1x, P1y, color="red")
    # P1xx = np.sin(wRad) * Radius
    # plt.plot([P1xx, P1xx], [0, 5])
    # LineI = P0y + slope * (wDeg - P0x)
    # x = np.cos(wRad) * Radius
    # y = np.sin(wRad) * Radius
    # plt.plot(x, y + np.sin(wRad) * Chip + Chip)
    # plt.close("all")

    plt.grid(True)
    plt.title(f"Model engagement comparison. Chipload: {Chip:<2.2f}")
    plt.xlabel("Degrees")
    plt.tight_layout()
    plt.show()
