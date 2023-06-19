import numpy as np
from scipy.signal import argrelmin, argrelmax
import logging
import argparse
import time as tm
import datetime
import string

from pathlib import Path
import multiprocessing
import traceback
from numba import jit


@jit(nopython=True)
def update_nutrient(matrix: np.array, phi: float, B: float, C: float, p: np.array, delta_t: float) -> float:
    new_phi = phi + (B * (1 - phi) - (matrix * p.reshape(len(p), 1)).sum() *
                          C * phi) * delta_t
    return new_phi


# @jit(nopython=True)
def death(matrix: np.array, damage_death_rate: np.array, B: float, delta_t: float) -> np.array:
    those_that_die = (damage_death_rate + B) * delta_t * matrix
    return those_that_die


@jit(nopython=True)
def grow(matrix: np.array, phi: float, A: float, r: float, E: float, p: np.array, delta_t: float,
         q: np.array) -> (np.array, np.array):
    those_that_grow = A * (1 - r / E) * phi * p.reshape(len(p), 1) * delta_t * matrix
    where_to_grow = np.concatenate((np.zeros_like(q).reshape((1, len(q))), those_that_grow[:-1, :]))
    return those_that_grow, where_to_grow


# @jit(nopython=True)
def accumulate_damage(matrix: np.array, D: float, F: float, delta_t: float, p: np.array, q: np.array
                      ) -> (np.array, np.array):
    F_prime = (1+F)**delta_t -1
    D_prime = D*len(q)
    those_that_accumulate = (np.zeros((len(p), len(q))) +
                             p.reshape(len(p), 1) * D_prime +
                             q.reshape(1, len(q)) * F_prime) * delta_t * matrix
    where_to_accumulate = np.concatenate((np.zeros_like(p).reshape((len(p), 1)),
                                          those_that_accumulate[:, :-1]), axis=1)
    return those_that_accumulate, where_to_accumulate


# @jit(nopython=True)
# def repair_damage(matrix: np.array, r: float, delta_t: float, p: np.array) -> np.array:
#     those_that_repair = np.concatenate((np.zeros_like(p).reshape((len(p), 1)),
#                                         (p * r * delta_t).reshape((len(p), 1)) * matrix[:, 1:]), axis=1)
#     where_to_repair = np.concatenate((those_that_repair[:, 1:],
#                                       np.zeros_like(p).reshape((len(p), 1))), axis=1)
#     return those_that_repair, where_to_repair

@jit(nopython=True)
def repair_damage(matrix: np.array, r: float, delta_t: float, p: np.array) -> np.array:
    r_prime = r * len(p)
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


logging.basicConfig(level=logging.INFO)


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
        self.p = np.linspace(1, 2, discretization_volume)
        self.q = np.linspace(0, 1, discretization_damage)
        self.rhos = np.outer(1 / self.p, self.q)
        self.damage_death_rate = (self.rhos / (1 - self.rhos)) ** self.params["G"]
        self.damage_death_rate[np.isinf(self.damage_death_rate)] = self.damage_death_rate[
            ~np.isinf(self.damage_death_rate)].max()
        self.delta_t = 1e-20

        # Initialize p x q matrix
        mean_p, mean_q, var_p, var_q, starting_phi, starting_popsize = \
            1 + np.random.random(),\
            np.random.random(), \
            np.random.random(), \
            np.random.random(), \
            np.random.random(), np.random.exponential(100000)
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

        self.time = 0
        self.history = History(self, save_path=save_path)
        self.converged = False
        self.max_delta_t = self.delta_t
        self.convergence_estimate_first_order = None
        self.convergence_estimate_second_order = None
        self.prev = 0
        self.prev_popsize = (self.matrix*self.rhos/self.matrix.sum()).sum()

    @staticmethod
    def alarm_matrix(matrix: np.array) -> None:
        if (matrix < 0).sum() > 0:
            logging.debug(f"{matrix} failed the check (matrix)")
            raise InvalidActionException

    @staticmethod
    def alarm_phi(phi: float) -> None:
        if phi < 0 or phi > 1:
            logging.debug(f"{phi} failed the check (nutrient)")
            raise InvalidActionException

    def check_convergence_v2(self):
        critical_period = self.max_delta_t*1000
        logging.info(f"checking convergence, critical period - {critical_period}")
        # Claiming convergence only if critical period of time passed
        # print(self.history.times[-1], critical_period, self.max_delta_t)
        if self.history.times[-1] > critical_period:
            logging.info("really checking convergence")
            ii = (-np.array(self.history.times) + self.history.times[-1]) < critical_period
            if len(set(np.round(np.array(self.history.population_sizes)[ii]))) == 1:
                # Last 'critical period' of time was with the same population size
                self.converged = True
                logging.info(f"same population size for {critical_period} time")
            else:
                minima, maxima, t_minima, t_maxima = self.history.get_peaks()
                minima, maxima, t_minima, t_maxima = minima[-min(len(minima), len(maxima)):], \
                    maxima[-min(len(minima), len(maxima)):], \
                    t_minima[-min(len(minima), len(maxima)):], \
                    t_maxima[-min(len(minima), len(maxima)):]
                if len(minima) >= 2 and len(maxima) >= 2: # If there were more than two minima and maxima
                    logging.info("convergence estimate could change now")
                    estimate = (minima[-1] + maxima[-1]) / 2 # Estimate based on last two 1st order peaks
                    if self.convergence_estimate_first_order is not None:
                          print("prev n of peaks", len(minima) + len(maxima), 'current n of peaks', self.convergence_estimate_first_order[1])
                    if self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period/4 and \
                            len(minima) + len(maxima) != self.convergence_estimate_first_order[1] and \
                            int(self.convergence_estimate_first_order[0]) == int(estimate):
                        self.converged = True
                        logging.info(
                            f"converged, same 1st order convergence estimate {estimate} as before: "
                            f"{self.convergence_estimate_first_order}")
                    # Else if there was no 1st order convergence estimate or
                    # there is one and some additional peaks arrived, update the 1st order convergence estimate
                    elif self.convergence_estimate_first_order is None or \
                            self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period/ 4 and \
                                    len(minima) + len(maxima) != self.convergence_estimate_first_order[1]:
                        self.convergence_estimate_first_order = [estimate, len(minima) + len(maxima), self.time]
                        logging.info(
                            f"changing 1st order convergence estimate: {self.convergence_estimate_first_order}")
                smoothed, t_smoothed = (minima + maxima) / 2, (t_minima + t_maxima) / 2
                if len(smoothed) > 1:
                    index_array = np.where(np.round(smoothed) != np.round(smoothed)[-1])[0]
                    if len(index_array) == 0:
                        last_time = t_smoothed[0]
                    else:
                        last_time = t_smoothed[np.max(index_array)+1]
                    if self.history.times[-1] - last_time > critical_period:
                        self.converged = True
                        logging.info(f"converged, same population size for {critical_period} time")
                smoothed_minima, smoothed_maxima = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
                if len(smoothed_minima) >= 2 and len(smoothed_maxima) >= 2:
                    logging.info("convergence estimate could change now")
                    estimate = (smoothed_minima[-1] + smoothed_maxima[-1])/2
                    if self.convergence_estimate_second_order is not None and \
                            len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1] and \
                            int(self.convergence_estimate_second_order[0]) == int(estimate):
                        self.converged = True
                        logging.info(
                            f"converged, same 2nd order convergence estimate {estimate} as before: {self.convergence_estimate_second_order}")
                    elif self.convergence_estimate_second_order is None or self.convergence_estimate_second_order is not None \
                            and len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1]:
                        self.convergence_estimate_second_order = [estimate, len(smoothed_minima) + len(smoothed_maxima)]
                        logging.info(f"changing 2nd order convergence estimate: {self.convergence_estimate_second_order}")

        # print("-------")
        # print("-------")
        # print("-------")
        # print("-------")

    # def check_convergence(self):
    #     logging.info("checking convergence")
    #     critical_period = self.max_delta_t*5000
    #     logging.info(f"{self.history.times[-1]}, {critical_period}")
    #     if self.history.times[-1] > critical_period:
    #         logging.info("really checking convergence")
    #         ii = (-np.array(self.history.times) + self.history.times[-1]) < critical_period
    #         if len(set(np.round(np.array(self.history.population_sizes)[ii]))) == 1:
    #             self.converged = True
    #             logging.info(f"same population size for {critical_period} time")
    #         else:
    #             minima, maxima, t_minima, t_maxima = self.history.get_peaks()
    #             minima, maxima, t_minima, t_maxima = minima[:min(len(minima), len(maxima))], \
    #                 maxima[:min(len(minima), len(maxima))], \
    #                 t_minima[:min(len(minima), len(maxima))], \
    #                 t_maxima[:min(len(minima), len(maxima))]
    #             if len(minima) >= 2 and len(maxima) >= 2:
    #                 logging.info("convergence estimate could change now")
    #                 estimate = (minima[-1] + maxima[-1])/2
    #                 if self.convergence_estimate_first_order is not None and \
    #                         self.time > self.convergence_estimate_first_order[2] + critical_period/4 and \
    #                         len(minima) + len(maxima) != self.convergence_estimate_first_order[1] and \
    #                         int(self.convergence_estimate_first_order[0]) == int(estimate):
    #                     self.converged = True
    #                     logging.info(
    #                         f"converged, same 1st order convergence estimate {estimate} as before: {self.convergence_estimate_first_order}")
    #                 elif self.convergence_estimate_first_order is None or \
    #                         self.convergence_estimate_first_order is not None and \
    #                         self.time > self.convergence_estimate_first_order[2] + critical_period/4 and \
    #                         len(minima) + len(maxima) != self.convergence_estimate_first_order[1]:
    #                     self.convergence_estimate_first_order = [estimate, len(minima) + len(maxima), self.time]
    #                     logging.info(f"changing 1st order convergence estimate: {self.convergence_estimate_first_order}")
    #             smoothed, t_smoothed = (minima + maxima)/2, (t_minima + t_maxima)/2
    #             if len(smoothed) > 1:
    #                 index_array = np.where(np.round(smoothed) != np.round(smoothed)[-1])[0]
    #                 if len(index_array) == 0:
    #                     last_time = t_smoothed[0]
    #                 else:
    #                     last_time = t_smoothed[np.max(index_array)+1]
    #                 if self.history.times[-1] - last_time > critical_period:
    #                     self.converged = True
    #                     logging.info(f"converged, same population size for {critical_period} time")
    #             smoothed_minima, smoothed_maxima = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
    #             logging.info(f"{minima[-10:]}, {maxima[-10:]}")
    #             logging.info(f"smoothed minima and maxima: {smoothed_minima}, {smoothed_maxima}")
    #             if len(smoothed_minima) >= 2 and len(smoothed_maxima) >= 2:
    #                 logging.info("convergence estimate could change now")
    #                 estimate = (smoothed_minima[-1] + smoothed_maxima[-1])/2
    #                 if self.convergence_estimate_second_order is not None and \
    #                         len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1] and \
    #                         int(self.convergence_estimate_second_order[0]) == int(estimate):
    #                     self.converged = True
    #                     logging.info(
    #                         f"converged, same 2nd order convergence estimate {estimate} as before: {self.convergence_estimate_second_order}")
    #                 elif self.convergence_estimate_second_order is None or self.convergence_estimate_second_order is not None \
    #                         and len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1]:
    #                     self.convergence_estimate_second_order = [estimate, len(smoothed_minima) + len(smoothed_maxima)]
    #                     logging.info(f"changing 2nd order convergence estimate: {self.convergence_estimate_second_order}")


    def death(self, matrix: np.array) -> np.array:
        those_that_die = (self.damage_death_rate + self.params["B"]) * self.delta_t * matrix
        return those_that_die

    def update_nutrient(self) -> float:
        new_phi = self.phi + (self.params["B"] * (1 - self.phi) - (self.matrix * self.p.reshape(len(self.p), 1)).sum() *
                              self.params["C"]) * self.delta_t
        return new_phi

    def grow(self, matrix: np.array, phi: float) -> (np.array, np.array):
        those_that_grow = self.params["A"] * (1 - self.params["r"] / self.params["E"]) * phi \
                          * self.p.reshape(len(self.p), 1) * self.delta_t * matrix
        where_to_grow = np.vstack([np.zeros_like(self.q).reshape((1, len(self.q))), those_that_grow[:-1, :]])
        return those_that_grow, where_to_grow

    def divide(self, matrix: np.array) -> (np.array, np.array, np.array):
        those_that_divide = matrix[-1, :]
        damage = np.arange(len(self.q))
        where_to_divide_1 = (damage * (1 - self.params["a"]) / 2).round().astype(int)
        where_to_divide_2 = damage - where_to_divide_1
        return those_that_divide, where_to_divide_1, where_to_divide_2

    def accumulate_damage(self, matrix: np.array) -> (np.array, np.array):
        F_prime = self.params["F"]**self.delta_t
        D_prime = self.params["D"]*len(self.q)
        those_that_accumulate = (np.zeros((len(self.p), len(self.q))) +
                                 self.p.reshape(len(self.p), 1) * D_prime +
                                 self.q.reshape(1, len(self.q)) * F_prime) * self.delta_t * matrix
        where_to_accumulate = np.hstack([np.zeros_like(self.p).reshape((len(self.p), 1)),
                                         those_that_accumulate[:, :-1]])
        return those_that_accumulate, where_to_accumulate

    def repair_damage(self, matrix: np.array) -> np.array:
        those_that_repair = (self.p * self.params["r"] * self.delta_t).reshape((len(self.p), 1)) * matrix
        where_to_repair = np.hstack([those_that_repair[:, :-1],
                                     np.zeros_like(self.p).reshape((len(self.p), 1))])
        return those_that_repair, where_to_repair

    def step(self, step_number: int):
        logging.debug(f"trying delta_t = {self.delta_t}")
        logging.debug(f"matrix at the start of the iteration:\n{self.matrix}")
        t0 = tm.time()
        new_phi = update_nutrient(self.matrix, self.phi, self.params["B"], self.params["C"], self.p,
                                  self.delta_t)
        t1 = tm.time()
        self.alarm_phi(new_phi)
        logging.debug("nutrient checked")
        t2 = tm.time()
        # print("nutrient: ", t2-t0)
        death_from = death(self.matrix, self.damage_death_rate, self.params["B"], self.delta_t)
        t3 = tm.time()
        # print("death: ", t3-t2)
        grow_from, grow_to = grow(self.matrix, self.phi, self.params["A"],
                                  self.params["r"], self.params["E"],
                                  self.p, self.delta_t, self.q)
        t4 = tm.time()
        # print("growth: ", t4-t3)
        accumulate_from, accumulate_to = accumulate_damage(self.matrix, self.params["D"],
                                                           self.params["F"], self.delta_t,
                                                           self.p, self.q)
        t5 = tm.time()
        # print("accumulate: ", t5-t4)
        repair_from, repair_to = repair_damage(self.matrix, self.params["r"], self.delta_t, self.p)
        t6 = tm.time()
        # print("repair", t6-t5)
        # print("total:", t6-t1)
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
        t7 = tm.time()
        # print("alarms: ", t7 - t6)
        new_matrix = self.matrix - death_from - grow_from + grow_to - accumulate_from + accumulate_to \
                     - repair_from + repair_to

        new_matrix = divide(new_matrix, self.q, self.params["a"])
        logging.debug("checking combination")
        self.alarm_matrix(new_matrix)
        logging.debug("combination checked")
        # print("division: ", tm.time() - t7)
        accept_step = True
        self.matrix = new_matrix
        self.phi = new_phi
        self.time += self.delta_t
        self.delta_t = self.delta_t * 2
        # self.delta_t = min(self.delta_t, 0.01)#, self.phi)
        self.max_delta_t = max(self.max_delta_t, self.delta_t)
        self.history.record()
        if step_number % 1000 == 0:
            logging.info(
                f"time = {self.time}, population size = {self.matrix.sum()}, delta_t: {self.delta_t}, phi={self.phi}")
            self.check_convergence_v2()
        t9 = tm.time()
        # print("super total: ", t9 - t0)
        # print("------------")
        # if self.time > self.prev + 1:
        #     logging.info(f"{self.time}, {self.matrix.sum()}")
        #     self.prev = self.time
        #     self.matrix /= self.matrix.sum()

        return accept_step

    def run(self, n_steps: int) -> None:
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

                        # break
                # if self.delta_t == 0 or self.converged:
                if self.converged:
                    break
                if self.mode == "interactive":
                    self.drawer.draw_step(step_number, self.delta_t)
        except Exception:
            error_message = traceback.format_exc()
            logging.error(error_message)
        finally:
            # save = self.converged if self.mode == "cluster" else True
            # if save:
            self.history.save()


class History:
    def __init__(self, simulation_obj: Simulation, save_path: str):
        self.simulation = simulation_obj
        self.population_sizes = []
        self.times = []
        self.real_times = []
        self.save_path = save_path
        Path(self.save_path).mkdir(exist_ok=True)
        self.starting_time = tm.time()

    def record(self) -> None:
        self.population_sizes.append(self.simulation.matrix.sum())
        self.times.append(self.simulation.time)
        self.real_times.append(tm.time() - self.starting_time)

    def get_peaks(self) -> (np.array, np.array, np.array, np.array):
        popsizes, times = np.array(self.population_sizes), np.array(self.times)
        minima, t_minima = popsizes[argrelmin(popsizes, order=10)], times[argrelmin(popsizes, order=10)]
        maxima, t_maxima = popsizes[argrelmax(popsizes, order=10)], times[argrelmax(popsizes, order=10)]
        return minima, maxima, t_minima, t_maxima

    def save(self):
        with open(f"{self.save_path}/population_size_history.txt", "w") as fl:
            fl.write(" ".join(list(map(str, self.times))) + '\n')
            fl.write(" ".join(list(map(str, self.population_sizes))) + '\n')
            fl.write(" ".join(list(map(str, self.real_times))) + '\n')
        with open(f"{self.save_path}/final_state.txt", "w") as fl:
            for el in self.simulation.matrix:
                fl.write(" ".join(map(str, el)) + '\n')
        with open(f"{self.save_path}/final_phi.txt", "w") as fl:
            fl.write(str(self.simulation.phi) + '\n')
        with open(f"{self.save_path}/simulation_length.txt", "w") as fl:
            fl.write(str(tm.time() - self.real_times[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator")
    parser.add_argument("-m", "--mode", default="local", type=str, choices=["cluster", "local", "interactive"])
    parser.add_argument("-ni", "--niterations", default=100000, type=int)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("-nt", "--nthreads", type=int, default=1)
    parser.add_argument("-np", "--nprocs", default=1, type=int)
    parser.add_argument("-A", type=float)  # nu * x * phi0
    parser.add_argument("-B", type=float)  # R / V
    parser.add_argument("-C", type=float)  # nu * x / V
    parser.add_argument("-D", type=float)  # d / K ?
    parser.add_argument("-E", type=float)  # 0 < E <= 1
    parser.add_argument("-F", type=float)  # 0 <= F
    parser.add_argument("-G", type=float)  # G
    parser.add_argument("-a", type=float)  # 0 <= a <= 1
    parser.add_argument("-r", type=float)  # 0 <= r <= E
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
                    f"{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}_{args.a}_{args.r}"
        n_procs = args.nprocs
        if args.mode == "interactive":
            from master_interactive_mode import Drawer
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}_{args.a}_{args.r}"
        n_procs = 1
    parameters = {"A": args.A, "B": args.B, "C": args.C, "D": args.D, "E": args.E, "F": args.F, "G": args.G,
                  "a": args.a, "r": args.r}
    run_id = str(round(datetime.datetime.now().timestamp() * 1000000))
    letter = np.random.choice(list(string.ascii_lowercase))
    # run_name = args.run_name if args.run_name != "" else "_" + "_".join([RandomWord().word(starts_with=letter,
    #                                                                                   include_parts_of_speech=[
    #                                                                                       "adjective"]),
    #                                                                 RandomWord().word(starts_with=letter,
    #                                                                                   include_parts_of_speech=[
    #                                                                                       "noun"])])
    # save_path = f"{save_path}/{run_id}{run_name}"
    Path(save_path).mkdir(exist_ok=True)
    if n_procs > 1:
        for i in tqdm(range(0, args.nthreads, n_procs)):
            processes = []
            simulations = []
            for j in range(i, min(i + n_procs, args.nthreads)):
                existing_folders = list(map(int, [file.stem for file in Path(save_path).glob("*")]))
                current_folder = Path(f"{save_path}/{max(existing_folders) + 1 if existing_folders else 1}")
                current_folder.mkdir()
                simulation = Simulation(params=parameters, mode=args.mode,
                                        save_path=str(current_folder) if args.save_path is None else args.save_path,
                                        discretization_volume=args.discretization_volume,
                                        discretization_damage=args.discretization_damage
                                        )
                simulations.append(simulation)
                processes.append(multiprocessing.Process(target=simulation.run,
                                                         args=[args.niterations]))
            for process in processes:
                process.start()
            for process in processes:
                process.join()
    else:
        for _ in range(args.nthreads):
            existing_folders = list(map(int, [file.stem for file in Path(save_path).glob("*")]))
            current_folder = Path(f"{save_path}/{max(existing_folders) + 1 if existing_folders else 1}")
            current_folder.mkdir()
            print(parameters)
            simulation = Simulation(params=parameters, mode=args.mode,
                                    save_path=str(current_folder) if args.save_path is None else args.save_path,
                                    discretization_volume=args.discretization_volume,
                                    discretization_damage=args.discretization_damage)

            simulation.run(args.niterations)

