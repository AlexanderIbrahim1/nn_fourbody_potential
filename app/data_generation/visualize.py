"""
This module contains functions to help visualize the four points in 3D space.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from common_types import FourCartesianPoints


AxisLimits = tuple[float, float]


def visualize_in_3d(points: FourCartesianPoints) -> None:
    # unpack the points into x, y, z coordinates
    xdata = [points[i][0] for i in range(4)]
    ydata = [points[i][1] for i in range(4)]
    zdata = [points[i][2] for i in range(4)]

    # create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot the points
    ax.scatter(xdata, ydata, zdata, c="r", marker="o")

    # draw lines between each pair of points
    for point1, point2 in itertools.combinations(points, 2):
        xs, ys, zs = zip(point1, point2)
        ax.plot(xs, ys, zs, "b")

    # make the aspect ratio equal for all 3 dimensions
    ax.set_box_aspect((np.ptp(xdata), np.ptp(ydata), np.ptp(zdata)))

    # label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # show plot
    plt.show()
