import numpy as np
from numba import jit


@jit(nopython=True)
def update_nutrient(matrix: np.array, phi: float, B: float, C: float, p: np.array, delta_t: float) -> float:
    # TESTED
    new_phi = phi + (B * (1 - phi) - (matrix * p.reshape(len(p), 1)).sum() *
                     C * phi) * delta_t
    return new_phi


@jit(nopython=True)
def death(matrix: np.array, damage_death_rate: np.array, B: float, delta_t: float) -> np.array:
    # TESTED
    those_that_die_from_dilution = B * delta_t * matrix
    those_that_die_from_damage = damage_death_rate * delta_t * matrix
    dead = those_that_die_from_dilution + those_that_die_from_damage
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] < 1e-50 and dead[i, j] > matrix[i, j]:
                dead[i, j] = matrix[i, j]
    return dead


@jit(nopython=True)
def grow(matrix: np.array, phi: float, A: float, r: float, E: float, p: np.array, delta_t: float,
         q: np.array) -> (np.array, np.array):
    # TESTED
    those_that_grow = A * (1 - r / E) * phi * p.reshape(len(p), 1) * delta_t * matrix
    where_to_grow = np.concatenate((np.zeros_like(q).reshape((1, len(q))), those_that_grow[:-1, :]))
    return those_that_grow, where_to_grow


def accumulate_damage(matrix: np.array, D: float, F: float, delta_t: float,
                      p: np.array, q: np.array
                      ) -> (np.array, np.array):
    # TESTED
    those_that_accumulate = (np.zeros((len(p), len(q))) +
                             p.reshape(len(p), 1) * D * len(q) +
                             q.reshape(1, len(q)) * F) * delta_t * matrix
    where_to_accumulate = np.concatenate((np.zeros_like(p).reshape((len(p), 1)),
                                          those_that_accumulate[:, :-1]), axis=1)
    return those_that_accumulate, where_to_accumulate


@jit(nopython=True)
def repair_damage(matrix: np.array, r: float, delta_t: float, p: np.array, q: np.array) -> np.array:
    # TESTED
    those_that_repair = (p * r * len(q) * delta_t).reshape((len(p), 1)) * matrix
    those_that_repair[:, 0] = 0
    where_to_repair = np.concatenate((those_that_repair[:, 1:],
                                      np.zeros_like(p).reshape((len(p), 1))), axis=1)
    return those_that_repair, where_to_repair


@jit(nopython=True)
def divide(matrix: np.array, q: np.array, a: float) -> (np.array, np.array, np.array):
    those_that_divide = matrix[-1, :]
    damage = np.arange(len(q))
    where_to_divide_1 = damage * (1 - a) / 2
    where_to_divide_1 = np.array([int(el) for el in where_to_divide_1])
    where_to_divide_2 = damage - where_to_divide_1
    for k in range(len(where_to_divide_1)):
        matrix[0, where_to_divide_1[k]] += those_that_divide[k]

    for k in range(len(where_to_divide_2)):
        matrix[0, where_to_divide_2[k]] += those_that_divide[k]

    matrix[-1, :] -= those_that_divide
    return matrix


def clear_nonexistent(matrix: np.array, rhos: np.array):
    matrix[rhos >= 0.97] = 0
    return matrix
