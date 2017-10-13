"""
generate test matrices for cross-associations
"""

import numpy as np


def generate_matrix(num_cols, num_rows, noise=0, num_patches=0):
    """
    Generate a matrix for testing
    :param num_cols: number of columns
    :param num_rows: number of rows
    :param noise: random noise, in (0, 1)
    :param num_patches: how many "patches" to add
    """
    random_matrix = np.random.binomial(1, noise, size=(num_rows, num_cols))

    if num_patches:
        x_size = num_cols // num_patches
        x_off = [_ for _ in range(0, num_cols - num_cols %
                                  num_patches, x_size)]

        y_size = num_cols // num_patches
        y_off = [_ for _ in range(0, num_rows - num_rows %
                                  num_patches, y_size)]

        for i, x_val in enumerate(x_off):
            x_begin = x_val
            x_end = x_val + x_size
            y_begin = y_off[i]
            y_end = y_off[i] + y_size
            random_matrix[x_begin:x_end, y_begin:y_end] = 1

    random_matrix = random_matrix.transpose()
    np.random.shuffle(random_matrix)
    random_matrix.transpose()
    np.random.shuffle(random_matrix)

    return random_matrix
