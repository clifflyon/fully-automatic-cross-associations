"""
plotting functions

* plot shaded: show a summary of density for each cluster

"""

from collections import Counter

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# from drawnow import drawnow, figure


def get_density(matrix, q_col, q_row, col_cluster, row_cluster):
    """
    Compute the density of a patch

    :param M: input matrix
    :param Q_col: column cluster memberships
    :param Q_row: row cluster memberships
    :param col_cluster: the row cluster
    :param row_cluster: the column cluster

    """
    ones = 0
    total = 0
    for col in [_ for _, i in enumerate(q_col) if i == col_cluster]:
        for row in [_ for _, i in enumerate(q_row) if i == row_cluster]:
            ones += matrix[row, col]
            total += 1
    assert total > 0
    return ones * 1.0 / total


def plot_spy(matrix_object, markersize=2, color="blue"):
    """
    Plot sparse matrix like matlab

    :param matrix_object: matrix to plot
    :param markersize: how big is each data point
    :param color: what color is each point
    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    ax1.spy(matrix_object.transformed_matrix,
            markersize=markersize, color=color)
    plt.show()


def plot_spy_original(matrix_object, markersize=2, color="blue"):
    """
    Plot sparse matrix like matlab

    :param matrix_object: matrix to plot
    :param markersize: how big is each data point
    :param color: what color is each point
    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.spy(matrix_object.matrix, markersize=markersize, color=color)
    plt.show()


def plot_shaded(matrix_object):
    """
    plot greyscale density plot

    :param matrix_object: binary matrix with cluster attributes
    """
    col_counter = Counter(matrix_object.col_clusters)
    row_counter = Counter(matrix_object.row_clusters)
    height, width = np.shape(matrix_object.matrix)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(right=width)
    ax1.set_ylim(top=height)
    # ax1.set_aspect("equal")
    col_offset = 0
    for col, col_len in col_counter.most_common():
        row_offset = 0
        for row, row_len in row_counter.most_common():
            density = get_density(matrix_object._dok_copy,
                                  matrix_object.col_clusters,
                                  matrix_object.row_clusters,
                                  col, row)
            ax1.add_patch(
                patches.Rectangle(
                    (col_offset * 1.0, row_offset * 1.0),
                    (col_offset + col_len) * 1.0,
                    (row_offset + row_len) * 1.0,
                    facecolor=str(1.0 - density),
                    edgecolor="#0000FF",
                    linewidth=1
                )
            )
            row_offset += row_counter[row]
        col_offset += col_counter[col]
    ax1.invert_yaxis()
    plt.show()
