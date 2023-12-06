import numpy as np
from numba import jit


@jit(nopython=True)
def divide(matrix: np.array, q: np.array, nondivision_threshold: int) -> np.array:
    those_that_divide = matrix[-1, :].copy()
    those_that_divide[nondivision_threshold:] = 0
    damage = np.arange(len(q))
    where_to_divide_1 = damage / 2
    where_to_divide_1 = np.array([int(el) for el in where_to_divide_1])
    where_to_divide_2 = damage - where_to_divide_1
    for k in range(len(where_to_divide_1)):
        matrix[0, where_to_divide_1[k]] += those_that_divide[k]

    for k in range(len(where_to_divide_2)):
        matrix[0, where_to_divide_2[k]] += those_that_divide[k]

    matrix[-1, :] -= those_that_divide
    return matrix

