import atexit

import numpy as np
import pandas as pd
from scipy.signal import argrelmin, argrelmax
import logging
import argparse
import time as tm

from pathlib import Path
import traceback
from numba import jit
import warnings


def get_peaks(lst: list):
    peaks = np.array(lst)[sorted(list(argrelmin(np.array(lst))[0]) + list(argrelmax(np.array(lst))[0]))]
    return peaks


def peak_distance(peaks):
    return np.abs(np.ediff1d(peaks))


def peak_distance_dynamics(peaks):
    peak_d = peak_distance(peaks)
    return np.ediff1d(peak_d)


def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


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
        return (peaks[-1] + peaks[-2])/2
    elif len(peaks) == 1:
        return peaks[0]


@jit(nopython=True)
def update_nutrient(matrix: np.array, phi: float, B: float, C: float, p: np.array, delta_t: float) -> float:
    new_phi = phi + (B * (1 - phi) - (matrix * p.reshape(len(p), 1)).sum() *
                          C * phi) * delta_t
    return new_phi


@jit(nopython=True)
def update_phage(matrix: np.array,
                 damage_death_rate: np.array,
                 ksi: float, B: float, C: float, F: float, H: float, p: np.array, q: np.array, delta_t: float) -> float:
    diluted = B * ksi * delta_t
    sucked_by_cells = C * ksi * (matrix * p.reshape(len(p), 1)).sum() * delta_t
    exiting_from_cells_by_death = H * (damage_death_rate * matrix * q.reshape(1, len(q))).sum() * delta_t
    exiting_from_cells_by_accumulation = H * \
                                          ((matrix*(np.zeros((len(p), len(q))) +
                                                     p.reshape(len(p), 1) * C * ksi +
                                                     q.reshape(1, len(q)) * F))[:, -1].sum() * q[-1]) * delta_t
    new_ksi = ksi - diluted - sucked_by_cells + exiting_from_cells_by_death + exiting_from_cells_by_accumulation
    return new_ksi


# @jit(nopython=True)
def death(matrix: np.array, damage_death_rate: np.array, B: float, delta_t: float) -> np.array:
    those_that_die_from_dilution = B * delta_t * matrix
    those_that_die_from_damage = damage_death_rate * delta_t * matrix
    burst_size = ((matrix.sum(axis=0) / matrix.sum()) *
                  np.arange(matrix.shape[1])).sum()
    return those_that_die_from_dilution + those_that_die_from_damage, burst_size


@jit(nopython=True)
def grow(matrix: np.array, phi: float, A: float, r: float, E: float, p: np.array, delta_t: float,
         q: np.array) -> (np.array, np.array):
    those_that_grow = A * (1 - r / E) * phi * p.reshape(len(p), 1) * delta_t * matrix
    where_to_grow = np.concatenate((np.zeros_like(q).reshape((1, len(q))), those_that_grow[:-1, :]))
    return those_that_grow, where_to_grow


# @jit(nopython=True)
def accumulate_damage(matrix: np.array, C: float, D: float, F: float, H: float,
                      ksi: float, delta_t: float,
                      p: np.array, q: np.array
                      ) -> (np.array, np.array):
    D_prime = D * len(q)
    if H > 0:
        D_prime += ksi*C
    those_that_accumulate = (np.zeros((len(p), len(q))) +
                             p.reshape(len(p), 1) * D_prime +
                             q.reshape(1, len(q)) * F) * delta_t * matrix
    where_to_accumulate = np.concatenate((np.zeros_like(p).reshape((len(p), 1)),
                                          those_that_accumulate[:, :-1]), axis=1)
    return those_that_accumulate, where_to_accumulate


@jit(nopython=True)
def repair_damage(matrix: np.array, r: float, delta_t: float, p: np.array, q: np.array) -> np.array:
    r_prime = r * len(q)
    those_that_repair = (p * r_prime * delta_t).reshape((len(p), 1)) * matrix
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


logging.basicConfig(level=logging.WARNING)


def gaussian_2d(x, y, mean_x, mean_y, var_x, var_y):
    return np.exp(-(np.power(x - mean_x, 2) / (2 * var_x) + np.power(y - mean_y, 2) / (2 * var_y)))


class InvalidActionException(Exception):
    pass


class OverTimeException(Exception):
    pass


class Simulation:
    def __init__(self,
                 params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251):
        self.mode = mode
        if self.mode == "interactive":
            self.drawer = Drawer(self)

        self.params = params
        # TODO
        self.params["F"] /= 500
        self.p = np.linspace(1, 2, discretization_volume)
        self.q = np.arange(discretization_damage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rhos = np.outer(1 / self.p, self.q/(len(self.q)-1))
            self.damage_death_rate = (self.rhos / (1 - self.rhos)) ** self.params["G"]
            self.damage_death_rate[np.isinf(self.damage_death_rate)] = self.damage_death_rate[
                ~np.isinf(self.damage_death_rate)].max()
        self.delta_t = 1e-20

        # Initialize p x q matrix
        mean_p, mean_q, var_p, var_q, starting_phi, starting_ksi, starting_popsize = \
            1 + np.random.random(),\
            np.random.random(), \
            np.random.random(), \
            np.random.random(), \
            np.random.random(), \
            np.random.random(), \
            np.random.exponential(100000)
        x, y = np.meshgrid(self.p, self.q)
        self.matrix = gaussian_2d(x.T, y.T, mean_p, mean_q, var_p, var_q)
        self.matrix = self.matrix/self.matrix.sum() * starting_popsize
        # self.matrix = np.zeros((len(self.p), len(self.q)))
        # self.matrix[:, 0] = 1/len(self.p)
        # # with open("/home/blackstone/PycharmProjects/AsymmetricDivisionsSimulations/simulations/data/master_equation/1685131265447297_knowledgeable_kingfish/1/final_state.txt", "r") as fl:
        #     mtx = []
        #     for line in fl.readlines():
        #         mtx.append(list(map(float, line.strip().split())))
        #     self.matrix = np.array(mtx)
        self.phi = starting_phi
        self.ksi = starting_ksi

        self.time = 0
        self.history = History(self, save_path=save_path)
        self.converged = False
        self.max_delta_t = self.delta_t
        self.convergence_estimate_first_order = None
        self.convergence_estimate_second_order = None
        self.convergence_estimate = None
        self.prev = 0
        self.prev_popsize = (self.matrix*self.rhos/self.matrix.sum()).sum()
        self.burst_size = 0

    @staticmethod
    def alarm_matrix(matrix: np.array) -> None:
        if (matrix < 0).sum() > 0:
            logging.debug(f"{matrix} failed the check (matrix)")
            raise InvalidActionException

    @staticmethod
    def alarm_phi(scalar: float) -> None:
        if scalar < 0 or scalar > 1:
            logging.debug(f"{scalar} failed the check (phi)")
            raise InvalidActionException

    @staticmethod
    def alarm_ksi(scalar: float) -> None:
        if scalar < 0:
            logging.debug(f"{scalar} failed the check (ksi)")
            raise InvalidActionException

    def check_convergence_v2(self):
        critical_period = self.max_delta_t*10000
        # Claiming convergence only if critical period of time passed
        if self.history.times[-1] > critical_period:
            ii = (-np.array(self.history.times) + self.history.times[-1]) < critical_period
            if len(set(np.round(np.array(self.history.population_sizes)[ii]))) == 1 and len(np.round(np.array(self.history.population_sizes)[ii])) > 1:
                # Last 'critical period' of time was with the same population size
                self.converged = True
                self.convergence_estimate = self.matrix.sum()
                logging.info(f"same population size for {critical_period} time")
            else:
                minima, maxima, t_minima, t_maxima = self.history.get_peaks()
                minima, maxima, t_minima, t_maxima = minima[-min(len(minima), len(maxima)):], \
                    maxima[-min(len(minima), len(maxima)):], \
                    t_minima[-min(len(minima), len(maxima)):], \
                    t_maxima[-min(len(minima), len(maxima)):]
                if len(minima) >= 2 and len(maxima) >= 2: # If there were more than two minima and maxima
                    estimate = (minima[-1] + maxima[-1]) / 2 # Estimate based on last two 1st order peaks
                    # if self.convergence_estimate_first_order is not None:
                    #       print("prev n of peaks", len(minima) + len(maxima), 'current n of peaks', self.convergence_estimate_first_order[1])
                    if self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period/4 and \
                            len(minima) + len(maxima) != self.convergence_estimate_first_order[1] and \
                            int(self.convergence_estimate_first_order[0]) == int(estimate):
                        self.converged = True
                        self.convergence_estimate = self.convergence_estimate_first_order
                        logging.info(
                            f"converged, same 1st order convergence estimate {estimate} as before: "
                            f"{self.convergence_estimate_first_order}")
                    # Else if there was no 1st order convergence estimate or
                    # there is one and some additional peaks arrived, update the 1st order convergence estimate
                    elif self.convergence_estimate_first_order is None or \
                            self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period / 4 and \
                                    len(minima) + len(maxima) != self.convergence_estimate_first_order[1]:
                        self.convergence_estimate_first_order = [estimate, len(minima) + len(maxima), self.time]
                        # logging.info(
                        #     f"changing 1st order convergence estimate: {self.convergence_estimate_first_order}")
                smoothed, t_smoothed = (minima + maxima) / 2, (t_minima + t_maxima) / 2
                if len(smoothed) > 5:
                    index_array = np.where(np.round(smoothed) != np.round(smoothed)[-1])[0]
                    if len(index_array) == 0:
                        last_time = t_smoothed[0]
                    else:
                        last_time = t_smoothed[np.max(index_array)+1]
                    if self.history.times[-1] - last_time > critical_period:
                        self.converged = True
                        self.convergence_estimate = self.matrix.sum()
                        logging.info(f"converged, same population size for {critical_period} time")
                smoothed_minima, smoothed_maxima = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
                if len(smoothed_minima) >= 2 and len(smoothed_maxima) >= 2:
                    estimate = (smoothed_minima[-1] + smoothed_maxima[-1])/2
                    if self.convergence_estimate_second_order is not None and \
                            len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1] and \
                            int(self.convergence_estimate_second_order[0]) == int(estimate):
                        self.converged = True
                        self.convergence_estimate = self.convergence_estimate_second_order
                        logging.info(
                            f"converged, same 2nd order convergence estimate {estimate} as before: {self.convergence_estimate_second_order}")
                    elif self.convergence_estimate_second_order is None or self.convergence_estimate_second_order is not None \
                            and len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1]:
                        self.convergence_estimate_second_order = [estimate, len(smoothed_minima) + len(smoothed_maxima)]
                        # logging.info(f"changing 2nd order convergence estimate: {self.convergence_estimate_second_order}")
                peaks = get_peaks(self.history.population_sizes)
                if convergence(peaks) == "cycle":
                    self.converged = True
                    self.convergence_estimate = equilibrium_N(peaks)
                    logging.info("got a cycle")

    def step(self, step_number: int):
        #self.delta_t = 0.001
        logging.debug(f"trying delta_t = {self.delta_t}")
        logging.debug(f"matrix at the start of the iteration:\n{self.matrix}")
        new_phi = update_nutrient(self.matrix, self.phi, self.params["B"], self.params["C"], self.p,
                                  self.delta_t)

        self.alarm_phi(new_phi)

        if self.params["H"] > 0:
            new_ksi = update_phage(self.matrix,
                                   self.damage_death_rate,
                                   self.ksi,
                                   self.params["B"], self.params["C"], self.params["F"], self.params["H"],
                                   self.p, self.q,
                                   self.delta_t)


        logging.debug("nutrient checked")
        death_from, self.burst_size = death(self.matrix, self.damage_death_rate, self.params["B"], self.delta_t)
        grow_from, grow_to = grow(self.matrix, self.phi, self.params["A"],
                                  self.params["r"], self.params["E"],
                                  self.p, self.delta_t, self.q)
        accumulate_from, accumulate_to = accumulate_damage(self.matrix, self.params["C"], self.params["D"],
                                                           self.params["F"], self.params["H"],
                                                           self.ksi, self.delta_t,
                                                           self.p, self.q)
        repair_from, repair_to = repair_damage(self.matrix, self.params["r"], self.delta_t, self.p, self.q)
        logging.debug("checking death")
        # self.alarm_matrix(self.matrix - death_from)
        logging.debug("death checked")
        logging.debug("checking growth")
        # self.alarm_matrix(self.matrix - grow_from)
        logging.debug("growth checked")
        logging.debug("checking accumulation")
        # self.alarm_matrix(self.matrix - accumulate_from)
        logging.debug("accumulation checked")
        logging.debug("checking repair")
        # self.alarm_matrix(self.matrix - repair_from)
        logging.debug("repair checked")
        new_matrix = self.matrix - death_from - grow_from + grow_to - accumulate_from + accumulate_to \
                     - repair_from + repair_to

        new_matrix = divide(new_matrix, self.q, self.params["a"])
        logging.debug("checking combination")
        self.alarm_matrix(new_matrix)
        logging.debug("combination checked")
        accept_step = True
        self.matrix = new_matrix
        self.phi = new_phi
        if self.params["H"] > 0:
            self.ksi = new_ksi
        self.time += self.delta_t
        self.max_delta_t = max(self.max_delta_t, self.delta_t)
        if step_number % 2000 == 0:
            self.history.record()
            logging.info(
                f"time = {self.time}, population size = {self.matrix.sum()}, delta_t: {self.delta_t}, phi={self.phi}, "
                f"ksi={self.ksi}")
            self.check_convergence_v2()
        self.delta_t *= 2
        self.delta_t = min(self.delta_t, 0.01)
        # self.delta_t = min(self.delta_t, 0.01)#, self.phi)

        # if self.time > self.prev + 1:
        #     logging.info(f"{self.time}, {self.matrix.sum()}")
        #     self.prev = self.time
        #     self.matrix /= self.matrix.sum()

        return accept_step

    def run(self, n_steps: int, save=True) -> (np.array, float):
        starting_time = tm.time()
        max_time = 60*10
        try:
            if self.mode in ["local", "interactive"]:
                iterator = tqdm(range(n_steps))
            else:
                iterator = range(n_steps)
            for step_number in iterator:
                accept_step = False
                while not accept_step:
                    try:
                        accept_step = self.step(step_number)
                        if tm.time() > starting_time + max_time:
                            raise OverTimeException
                    except InvalidActionException:
                        if self.delta_t < 1e-300:
                            self.phi *= 1e10
                            self.matrix *= 0.9
                            print(self.phi, self.matrix.sum())
                            self.delta_t = 1e-20

                        self.delta_t /= 10
                    if self.delta_t == 0:
                        logging.warning("No way to make the next step")
                        self.delta_t = 1e-20
                # if self.delta_t == 0 or self.converged:
                if self.converged:
                    break
                if self.mode == "interactive":
                    self.drawer.draw_step(step_number, self.delta_t)
        except Exception:
            error_message = traceback.format_exc()
            logging.error(error_message)
        finally:
            self.history.record()
            if save:
                self.history.save()
        return self.matrix, self.phi


class History:
    def __init__(self, simulation_obj: Simulation, save_path: str):
        self.simulation = simulation_obj
        self.population_sizes = []
        self.times = []
        self.real_times = []
        self.save_path = save_path
        Path(self.save_path).mkdir(exist_ok=True)
        self.starting_time = tm.time()
        self.phage_history = []
        self.burst_sizes = []

    def record(self) -> None:
        self.population_sizes.append(self.simulation.matrix.sum())
        self.times.append(self.simulation.time)
        self.real_times.append(tm.time() - self.starting_time)
        if self.simulation.params["H"] > 0:
            self.phage_history.append(self.simulation.ksi)
            self.burst_sizes.append(self.simulation.burst_size)

    def get_peaks(self) -> (np.array, np.array, np.array, np.array):
        popsizes, times = np.array(self.population_sizes), np.array(self.times)
        minima, t_minima = popsizes[argrelmin(popsizes)], times[argrelmin(popsizes)]
        maxima, t_maxima = popsizes[argrelmax(popsizes)], times[argrelmax(popsizes)]
        return minima, maxima, t_minima, t_maxima

    def save(self):
        print("-------------------saving-------------------------")
        logging.info("convergence estimate " + str(self.simulation.convergence_estimate))
        if self.simulation.convergence_estimate is None:
            peaks = get_peaks(self.population_sizes)
            if convergence(peaks) in ["converged", "cycle"]:
                convergence_estimate = equilibrium_N(peaks)
            else:
                convergence_estimate = self.simulation.matrix.sum()
        else:
            convergence_estimate = self.simulation.convergence_estimate
            if type(convergence_estimate) == list:
                convergence_estimate = convergence_estimate[0]
        peaks = get_peaks(self.population_sizes)
        estimated_mode = convergence(peaks)
        text = f"{self.simulation.params['a']},{self.simulation.params['r']}," \
               f"{convergence_estimate},{self.simulation.converged},{estimated_mode}"
        if self.simulation.params["H"] > 0:
            peaks = get_peaks(self.phage_history)
            if len(peaks) > 1:
                ksi = (peaks[-2] + peaks[-1]) / 2
            else:
                ksi = self.phage_history[-1]
            text += "," + str(round(ksi, 5))
            peaks = get_peaks(self.burst_sizes)
            if len(peaks) > 1:
                burst_size = (peaks[-2] + peaks[-1]) / 2
            else:
                burst_size = self.phage_history[-1]
            text += "," + str(round(burst_size, 5))

        with open(f"{self.save_path}/population_size_estimate.txt", "a") as fl:
            fl.write(text + "\n")
        with open(f"{self.save_path}/population_size_history_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
                  "w") as fl:
            if self.population_sizes[-1]  < 1 and all([x >= y for x, y in zip(self.population_sizes, self.population_sizes[1:])]):
                self.times = [self.times[0], self.times[-1]]
                self.population_sizes = [self.population_sizes[0], self.population_sizes[-1]]
            fl.write(",".join(list(map(str, self.times))) + '\n')
            fl.write(",".join(list(map(str, self.population_sizes))) + '\n')
            if self.simulation.params["H"] > 0:
                fl.write(",".join(list(map(str, self.phage_history))) + '\n')
                fl.write(",".join(list(map(str, self.burst_sizes))) + '\n')


def write_completion(save_path):
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator")
    parser.add_argument("-m", "--mode", default="local", type=str, choices=["cluster", "local", "interactive"])
    parser.add_argument("-ni", "--niterations", default=100000, type=int)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("-A", type=float)  # nu * x * phi0
    parser.add_argument("-B", type=float)  # R / V
    parser.add_argument("-C", type=float)  # nu * x / V
    parser.add_argument("-D", type=float)  # d / K ?
    parser.add_argument("-E", type=float)  # 0 < E <= 1
    parser.add_argument("-F", type=float)  # 0 <= F
    parser.add_argument("-G", type=float)  # G
    parser.add_argument("-H", type=float, default=0)
    parser.add_argument("-a", type=int)  # 0 <= a <= 1
    parser.add_argument("-r", type=int)  # 0 <= r <= E
    parser.add_argument("--discretization_volume", type=int, default=251)
    parser.add_argument("--discretization_damage", type=int, default=251)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG)

    if args.mode in ["local", "interactive"]:
        from tqdm import tqdm

        save_path = f"../data/master_equation/" \
                    f"{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}"
        if args.mode == "interactive":
            from master_interactive_mode import Drawer
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}"

    Path(save_path).mkdir(exist_ok=True)
    atexit.register(lambda: write_completion(save_path))
    matrix, phi = None, None
    if args.H == 0:
        max_r = min(args.D, args.E) if args.D != 0 else min(args.F/100, args.E)
    else:
        max_r = min(args.D, args.E) if args.D != 0 else min(args.F / 100, args.E)
        # Finding an appropriate r
        max_r_guesses = [min(args.F/100, args.E)]
        stop_guessing = False
        while not stop_guessing:
            print(max_r_guesses[-1])
            test_parameters = {"A": args.A, "B": args.B, "C": args.C,
                          "D": args.D, "E": args.E, "F": args.F,
                          "G": args.G, "H": args.H, "a": 0, "r": max_r_guesses[-1]}
            simulation = Simulation(params=test_parameters, mode=args.mode,
                                    save_path=str(save_path) if args.save_path is None else args.save_path,
                                    discretization_volume=args.discretization_volume,
                                    discretization_damage=args.discretization_damage)
            simulation.matrix /= simulation.matrix.sum()
            simulation.phi = 1
            simulation.run(args.niterations, save=False)
            a0_convergence = simulation.convergence_estimate
            if type(a0_convergence) == list:
                a0_convergence = a0_convergence[0]

            test_parameters = {"A": args.A, "B": args.B, "C": args.C,
                          "D": args.D, "E": args.E, "F": args.F,
                          "G": args.G, "H": args.H, "a": 1, "r": max_r_guesses[-1]}
            simulation = Simulation(params=test_parameters, mode=args.mode,
                                    save_path=str(save_path) if args.save_path is None else args.save_path,
                                    discretization_volume=args.discretization_volume,
                                    discretization_damage=args.discretization_damage)
            simulation.matrix /= simulation.matrix.sum()
            simulation.phi = 1
            simulation.run(args.niterations, save=False)
            a1_convergence = simulation.convergence_estimate
            if type(a1_convergence) == list:
                a1_convergence = a1_convergence[0]
            if int(a0_convergence) != int(a1_convergence):
                stop_guessing = True
            else:
                max_r_guesses.append(max_r_guesses[-1] / args.r)
            logging.info(str(max_r_guesses))
        if len(max_r_guesses) > 1:
            max_r = max_r_guesses[-2]
        else:
            max_r = max_r_guesses[-1]
    a_neutral = False
    for r in np.linspace(0, max_r , args.r):
        equilibria = []
        for a in np.linspace(0, 1, args.a):
            # Do not rerun already existing estimations
            estimates_file = (Path(save_path)/Path("population_size_estimate.txt"))
            if estimates_file.exists():
                print(str(estimates_file))
                estimates = pd.read_csv(str(estimates_file), sep=",", header=None)
                relevant_estimates = estimates.loc[(abs(estimates[0] - a) < 1e-5) & (abs(estimates[1] - r) < 1e-5), :]
                if len(relevant_estimates) > 0:
                    logging.info(f"skipping a={a}, r={r}, estimate already exists")
                    if list(relevant_estimates[3])[0] == True:
                        equilibria.append(list(relevant_estimates[2])[0])
                    continue

            parameters = {"A": args.A, "B": args.B, "C": args.C,
                          "D": args.D, "E": args.E, "F": args.F,
                          "G": args.G, "H": args.H, "a": round(a, 5), "r": round(r, 5)}
            simulation = Simulation(params=parameters, mode=args.mode,
                                    save_path=str(save_path) if args.save_path is None else args.save_path,
                                    discretization_volume=args.discretization_volume,
                                    discretization_damage=args.discretization_damage)
            if matrix is not None and phi is not None and matrix.sum() > 0:
                if matrix.sum() < 1:
                    matrix = matrix/matrix.sum()
                simulation.matrix = matrix
                simulation.phi = phi
            if args.H > 0:
                simulation.matrix /= simulation.matrix.sum()
                simulation.phi = 1
            logging.info(f"starting simulation with params: {parameters}")
            matrix, phi = simulation.run(args.niterations)
            convergence_estimate = simulation.convergence_estimate
            if type(convergence_estimate) == list:
                convergence_estimate = convergence_estimate[0]
            if type(convergence_estimate) == float:
                equilibria.append(round(convergence_estimate))
        a_neutral = len(set(equilibria)) == 1 and len(equilibria) == args.a
        if a_neutral:
            break
    df = pd.read_csv(f"{save_path}/population_size_estimate.txt", header=None)
    rr = np.linspace(0, args.D, args.r)
    if len(rr) > 1 and args.H == 0 and not a_neutral:
        r_step = rr[1] - rr[0]
        max_r = max(df[1])
        while len(df.loc[(df[1] == max(df[1])) & (df[2] > 1)]) > 0:
            r_step *= 2
            r = min(max_r + r_step, args.E)
            max_r = r
            equilibria = []
            for a in np.linspace(0, 1, args.a):
                parameters = {"A": args.A, "B": args.B, "C": args.C,
                              "D": args.D, "E": args.E, "F": args.F,
                              "G": args.G, "H": args.H, "a": round(a, 5), "r": round(r, 5)}
                simulation = Simulation(params=parameters, mode=args.mode,
                                        save_path=str(save_path) if args.save_path is None else args.save_path,
                                        discretization_volume=args.discretization_volume,
                                        discretization_damage=args.discretization_damage)
                if matrix is not None and phi is not None and matrix.sum() > 0:
                    if matrix.sum() < 1:
                        matrix = matrix/matrix.sum()
                    simulation.matrix = matrix
                    simulation.phi = phi
                if args.H > 0:
                    simulation.matrix /= simulation.matrix.sum()
                    simulation.phi = 1
                logging.info(f"starting simulation with params: {parameters}")
                matrix, phi = simulation.run(args.niterations)
                convergence_estimate = simulation.convergence_estimate
                if type(convergence_estimate) == list:
                    convergence_estimate = convergence_estimate[0]
                if type(convergence_estimate) == float:
                    equilibria.append(round(convergence_estimate))
            a_neutral = len(set(equilibria)) == 1 and len(equilibria) == args.a
            if a_neutral:
                break
            df = pd.read_csv(f"{save_path}/population_size_estimate.txt", header=None)
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")



