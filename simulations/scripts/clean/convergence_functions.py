import numpy as np
from scipy.signal import argrelmin, argrelmax


def get_peaks(lst: list):
    """Returns the minima and maxima in the order they occur in the list"""
    # TESTED
    peaks = np.array(lst)[sorted(list(argrelmin(np.array(lst))[0]) + list(argrelmax(np.array(lst))[0]))]
    return peaks


def peak_distance(peaks):
    """Returns distance between neighbouring elements in peaks"""
    # TESTED
    return np.abs(np.ediff1d(peaks))


def peak_distance_dynamics(peaks):
    """Returns changes of distances between consecutive peaks; positive if distance increases, negative if decreases"""
    # TESTED
    peak_d = peak_distance(peaks)
    return np.ediff1d(peak_d)


def strictly_increasing(L):
    # TESTED
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    # TESTED
    return all(x > y for x, y in zip(L, L[1:]))


def convergence(peaks):
    distance = peak_distance(peaks)
    dynamics = peak_distance_dynamics(peaks)
    if len(dynamics) > 0:
        if distance[-1] < 1 or (all([el <= 0 for el in dynamics[1:]]) and any([el < 0 for el in dynamics[1:]])):
            return "converged"
        if len(distance) >= 5 and strictly_increasing(distance[-10:]):
            return "diverging"
        if (len(distance) >= 10 and (len(set(distance[-3:])) == 1 or
                                     not strictly_increasing(distance[-10:]) and
                                     not strictly_decreasing(distance[-10:]))
                and distance[-1] >= 1):
            return "cycle"
        return "undefined"
    else:
        return "not converged"


def equilibrium_N(peaks):
    if len(peaks) == 0 or peaks[-1] < 1:
        return 0
    elif len(peaks) >= 2:
        return (peaks[-1] + peaks[-2]) / 2
    elif len(peaks) == 1:
        return peaks[0]


def equilibrium_N_phages(peaks):
    if len(peaks) == 0 or peaks[-1] < 1:
        return 0
    elif len(peaks) >= 2:
        return np.sqrt(peaks[-1] * peaks[-2])
    elif len(peaks) == 1:
        return peaks[0]