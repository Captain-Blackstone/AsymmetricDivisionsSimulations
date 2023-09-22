import numpy as np
from numba import jit


@jit(nopython=True)
def update_phage(matrix: np.array,
                 damage_death_rate: np.array,
                 ksi: float, B: float, C: float, F: float, p: np.array, q: np.array, delta_t: float,
                 exited_phages: float
                 ) -> float:
    # TESTED
    diluted = B * ksi * delta_t
    sucked_by_cells = C * ksi * (matrix * p.reshape(len(p), 1)).sum() * delta_t
    exiting_from_cells_by_death = (damage_death_rate * matrix * q.reshape(1, len(q))).sum() * delta_t
    exiting_from_cells_by_accumulation = ((matrix * (np.zeros((len(p), len(q))) +
                                                     p.reshape(len(p), 1) * C * ksi +
                                                     q.reshape(1, len(q)) * F))[:, -1].sum() * q[-1]) * delta_t

    new_ksi = (ksi - diluted - sucked_by_cells + exiting_from_cells_by_death + exiting_from_cells_by_accumulation +
               exited_phages)
    return new_ksi


def accumulate_phage(matrix: np.array,
                     C: float,
                     F: float,
                     ksi: float, delta_t: float,
                     p: np.array, q: np.array, D=0.0) -> (np.array, np.array):
    # TESTED
    those_that_accumulate = (np.zeros((len(p), len(q))) +
                             p.reshape(len(p), 1) * (ksi * C + D * len(q)) +
                             q.reshape(1, len(q)) * F) * delta_t * matrix
    where_to_accumulate = np.concatenate((np.zeros_like(p).reshape((len(p), 1)),
                                          those_that_accumulate[:, :-1]), axis=1)
    return those_that_accumulate, where_to_accumulate


def clear_nonexistent(matrix: np.array, rhos: np.array):
    q = np.arange(matrix.shape[1])
    exited_mtx = matrix.copy()
    exited_mtx[rhos < 0.97] = 0
    exited_phages = (exited_mtx * q.reshape((1, len(q)))).sum()
    matrix[rhos >= 0.97] = 0
    return matrix, exited_phages