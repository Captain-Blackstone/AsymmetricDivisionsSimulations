from master_equation_functions import *
from convergence_functions import *

import numpy as np
from scipy.signal import argrelmin, argrelmax

import time as tm
from pathlib import Path
import traceback
import warnings
import logging
from tqdm import tqdm
from master_interactive_mode_clean import Drawer


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

        self.params = params.copy()
        # TODO
        self.params["F"] /= 500
        self.p = np.linspace(1, 2, discretization_volume)
        self.q = np.arange(discretization_damage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rhos = np.outer(1 / self.p, 2*self.q/(len(self.q)-1))
            self.damage_death_rate = (self.rhos / (1 - self.rhos)) ** self.params["G"]
            self.damage_death_rate[self.rhos >= 1] = 0
            self.damage_death_rate[np.isinf(self.damage_death_rate)] = self.damage_death_rate[
                ~np.isinf(self.damage_death_rate)].max()

        self.delta_t = 1e-20

        # Initialize p x q matrix
        mean_p, mean_q, var_p, var_q, starting_phi, starting_popsize = \
            (1 + np.random.random(),
             np.random.random(),
             np.random.random(),
             np.random.random(),
             np.random.random(),
             np.random.exponential(100000))
        x, y = np.meshgrid(self.p, self.q)
        self.matrix = gaussian_2d(x.T, y.T, mean_p, mean_q, var_p, var_q)
        self.matrix = self.matrix / self.matrix.sum() * starting_popsize
        self.phi = starting_phi

        self.time = 0
        self.history = History(self, save_path=save_path)
        self.converged = False
        self.max_delta_t = self.delta_t
        self.convergence_estimate_first_order = None
        self.convergence_estimate_second_order = None
        self.convergence_estimate = None
        self.prev = 0
        self.prev_popsize = (self.matrix * self.rhos / self.matrix.sum()).sum()
        self.proposed_new_matrix = None
        self.proposed_new_phi = None

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

    def accumulate_damage(self):
        return accumulate_damage(self.matrix, self.params["D"],
                                 self.params["F"], self.delta_t,
                                 self.p, self.q)

    @property
    def get_logging_text(self):
        return f"time = {self.time}, population size = {self.matrix.sum()}, delta_t: {self.delta_t}, phi={self.phi}"

    def check_convergence_v2(self):
        critical_period = self.max_delta_t * 10000
        # Claiming convergence only if critical period of time passed
        if self.history.times[-1] > critical_period:
            ii = (-np.array(self.history.times) + self.history.times[-1]) < critical_period
            if len(set(np.round(np.array(self.history.population_sizes)[ii]))) == 1 and len(
                    np.round(np.array(self.history.population_sizes)[ii])) > 1:
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
                if len(minima) >= 2 and len(maxima) >= 2:  # If there were more than two minima and maxima
                    estimate = (minima[-1] + maxima[-1]) / 2  # Estimate based on last two 1st order peaks
                    if self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period / 4 and \
                            len(minima) + len(maxima) != self.convergence_estimate_first_order[1] and \
                            int(self.convergence_estimate_first_order[0]) == int(estimate):
                        self.converged = True
                        self.convergence_estimate = self.convergence_estimate_first_order[0]
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
                        last_time = t_smoothed[np.max(index_array) + 1]
                    if self.history.times[-1] - last_time > critical_period:
                        self.converged = True
                        self.convergence_estimate = self.matrix.sum()
                        logging.info(f"converged, same population size for {critical_period} time")
                smoothed_minima, smoothed_maxima = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
                if len(smoothed_minima) >= 2 and len(smoothed_maxima) >= 2:
                    estimate = (smoothed_minima[-1] + smoothed_maxima[-1]) / 2
                    if (self.convergence_estimate_second_order is not None and
                            len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1]
                            and
                            int(self.convergence_estimate_second_order[0]) == int(estimate)):
                        self.converged = True
                        self.convergence_estimate = self.convergence_estimate_second_order[0]
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

    def upkeep_after_step(self) -> None:
        self.matrix = self.proposed_new_matrix
        self.phi = self.proposed_new_phi
        self.time += self.delta_t
        self.max_delta_t = max(self.max_delta_t, self.delta_t)
        self.delta_t *= 2
        self.delta_t = min(self.delta_t, 0.01)

    def clear_nonexistent(self):
        self.proposed_new_matrix = clear_nonexistent(matrix=self.proposed_new_matrix, rhos=self.rhos)


    def step(self, step_number: int):
        # self.delta_t = 0.001
        logging.debug(f"trying delta_t = {self.delta_t}")
        logging.debug(f"matrix at the start of the iteration:\n{self.matrix}")
        self.proposed_new_phi = update_nutrient(matrix=self.matrix,
                                                phi=self.phi,
                                                B=self.params["B"],
                                                C=self.params["C"],
                                                p=self.p,
                                                delta_t=self.delta_t)

        self.alarm_phi(self.proposed_new_phi)
        logging.debug("nutrient checked")
        death_from = death(matrix=self.matrix,
                           damage_death_rate=self.damage_death_rate,
                           B=self.params["B"],
                           delta_t=self.delta_t)
        grow_from, grow_to = grow(matrix=self.matrix,
                                  phi=self.phi,
                                  A=self.params["A"],
                                  r=self.params["r"], E=self.params["E"],
                                  p=self.p, delta_t=self.delta_t, q=self.q)
        accumulate_from, accumulate_to = self.accumulate_damage()
        repair_from, repair_to = repair_damage(matrix=self.matrix,
                                               r=self.params["r"],
                                               delta_t=self.delta_t,
                                               p=self.p, q=self.q)

        self.proposed_new_matrix = self.matrix - death_from - grow_from + grow_to - accumulate_from + accumulate_to \
                     - repair_from + repair_to

        self.proposed_new_matrix = divide(matrix=self.proposed_new_matrix, q=self.q, a=self.params["a"])
        self.clear_nonexistent()
        logging.debug("checking combination")
        self.alarm_matrix(self.proposed_new_matrix)
        logging.debug("combination checked")
        accept_step = True
        return accept_step

    def run(self, n_steps: int, save=True) -> (np.array, float):
        self.last_record_n = self.matrix.sum()
        starting_time = tm.time()
        max_time = 60 * 20
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
                        self.delta_t /= 10
                    if self.delta_t == 0:
                        logging.warning("No way to make the next step")
                        self.delta_t = 1e-20
                self.upkeep_after_step()
                if abs(self.matrix.sum() - self.last_record_n) > self.last_record_n*0.1 or step_number % 5000 == 0:
                    if np.random.uniform() < 0.1:
                        self.last_record_n = self.matrix.sum()
                        self.history.record()
                        logging.info(self.get_logging_text)
                        self.check_convergence_v2()
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
        self.text = ""

    def record(self) -> None:
        self.population_sizes.append(self.simulation.matrix.sum())
        self.times.append(self.simulation.time)
        self.real_times.append(tm.time() - self.starting_time)

    def get_peaks(self) -> (np.array, np.array, np.array, np.array):
        popsizes, times = np.array(self.population_sizes), np.array(self.times)
        minima, t_minima = popsizes[argrelmin(popsizes)], times[argrelmin(popsizes)]
        maxima, t_maxima = popsizes[argrelmax(popsizes)], times[argrelmax(popsizes)]
        return minima, maxima, t_minima, t_maxima

    def prepare_to_save(self) -> None:
        logging.info("-------------------saving-------------------------")
        logging.info("convergence estimate " + str(self.simulation.convergence_estimate))
        if self.simulation.convergence_estimate is None:
            peaks = get_peaks(self.population_sizes)
            if convergence(peaks) in ["converged", "cycle"]:
                convergence_estimate = equilibrium_N(peaks)
            else:
                convergence_estimate = self.simulation.matrix.sum()
        else:
            convergence_estimate = self.simulation.convergence_estimate
        peaks = get_peaks(self.population_sizes)
        estimated_mode = convergence(peaks)
        self.text = f"{self.simulation.params['a']},{self.simulation.params['r']},"\
                    f"{convergence_estimate},{self.simulation.converged},{estimated_mode}"

    def save(self) -> None:
        self.prepare_to_save()
        with open(f"{self.save_path}/population_size_estimate.txt", "a") as fl:
            fl.write(self.text + "\n")
        with open(
                f"{self.save_path}/population_size_history_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
                "w") as fl:
            if self.population_sizes[-1] < 1 and all(
                    [x >= y for x, y in zip(self.population_sizes, self.population_sizes[1:])]):
                self.times = [self.times[0], self.times[-1]]
                self.population_sizes = [self.population_sizes[0], self.population_sizes[-1]]
            fl.write(",".join(list(map(str, self.times))) + '\n')
            fl.write(",".join(list(map(str, self.population_sizes))) + '\n')
