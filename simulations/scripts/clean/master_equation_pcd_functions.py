import numpy as np
from numba import jit
from master_equation_functions import death as basic_death
from master_equation_functions import divide as asymmetric_division


@jit(nopython=True)
def divide(matrix: np.array, q: np.array) -> np.array:
    return asymmetric_division(matrix=matrix, q=q, a=0)

