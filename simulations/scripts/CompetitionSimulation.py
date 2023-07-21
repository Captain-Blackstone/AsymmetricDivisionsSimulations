import random

import numpy as np
from scipy.signal import argrelmin, argrelmax
import logging
import time as tm
from pathlib import Path
import traceback
from numba import jit
from tqdm import tqdm
import warnings
from itertools import filterfalse


@jit(nopython=True)
def suggest_consumed_nutrient(matrix: np.array, phi: float, C: float, p: np.array, delta_t: float) -> float:
    return (matrix * p.reshape(len(p), 1)).sum() * C * phi * delta_t


def death(matrix: np.array, damage_death_rate: np.array, B: float, delta_t: float) -> np.array:
    those_that_die = (damage_death_rate + B) * delta_t * matrix
    return those_that_die


@jit(nopython=True)
def grow(matrix: np.array, phi: float, A: float, r: float, E: float, p: np.array, delta_t: float,
         q: np.array) -> (np.array, np.array):
    those_that_grow = A * (1 - r / E) * phi * p.reshape(len(p), 1) * delta_t * matrix
    where_to_grow = np.concatenate((np.zeros_like(q).reshape((1, len(q))), those_that_grow[:-1, :]))
    return those_that_grow, where_to_grow


def accumulate_damage(matrix: np.array, D: float, F: float, delta_t: float, p: np.array, q: np.array
                      ) -> (np.array, np.array):
    F_prime = F  # ((1+F)**delta_t - 1)

    D_prime = D*len(q)
    those_that_accumulate = (np.zeros((len(p), len(q))) +
                             p.reshape(len(p), 1) * D_prime +
                             q.reshape(1, len(q)) * F_prime) * delta_t * matrix
    where_to_accumulate = np.concatenate((np.zeros_like(p).reshape((len(p), 1)),
                                          those_that_accumulate[:, :-1]), axis=1)
    return those_that_accumulate, where_to_accumulate


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
                 params_list: list,
                 population_sizes_list: list,
                 chemostat_params: dict,
                 deterministic_threshold: int,
                 mode: str,
                 # save_path: str,
                 discretization_volume: int,
                 discretization_damage: int):
        self.chemostat_params = chemostat_params
        self.subpopulations = [SubpopulationManager(simulation=self,
                                                    params=params,
                                                    deterministic_threshold=deterministic_threshold,
                                                    starting_population_size=population_size,
                                                    discretization_volume=discretization_volume,
                                                    discretization_damage=discretization_damage)
                               for params, population_size in zip(params_list, population_sizes_list)]
        self.mode = mode
        self.phi = np.random.random()
        self.time = 0
        self.delta_t = 1e-2
        # self.history = History(self, save_path=save_path)
        self.converged = False
        self.accept_step = False

    def alarm_scalar(self, scalar: float, check_name: str) -> None:
        if scalar < 0 or scalar > 1:
            logging.debug(f"{scalar} failed the check ({check_name})")
            self.accept_step = False

    def step(self):
        self.delta_t = min(self.delta_t*10, 0.01)
        self.accept_step = False
        suggested_new_phi = None
        while not self.accept_step:
            for subpopulation in self.subpopulations:
                subpopulation.suggest_step()
            self.accept_step = all([subpopulation.current_subpopulation.accept_step for subpopulation in self.subpopulations])
            suggested_new_phi = self.phi + self.chemostat_params["B"] * (1 - self.phi)*self.delta_t - \
                                sum([subpopulation.consumed_nutrient_concentration
                                                for subpopulation in self.subpopulations])
            self.alarm_scalar(suggested_new_phi, "phi")
            if not self.accept_step:
                self.delta_t /= 10
        for subpopulation in self.subpopulations:
            subpopulation.step()
        self.phi = suggested_new_phi

    def run(self, n_steps: int) -> None:
        starting_time = tm.time()
        max_time = 60*10
        try:
            if self.mode in ["local", "interactive"]:
                iterator = tqdm(range(n_steps))
            else:
                iterator = range(n_steps)
            for step_number in iterator:
                self.step()
                if tm.time() - starting_time > max_time:
                    raise OverTimeException
                if self.converged:
                    break
                if self.mode == "interactive":
                    self.drawer.draw_step(step_number, self.delta_t)
                if step_number % 100 == 0:
                    logging.info(f"population size: {self.population_size}, nutrient concentration: {self.phi}")
        except Exception:
            error_message = traceback.format_exc()
            logging.error(error_message)
        finally:
            # self.history.record()
            # self.history.save()
            pass

    @property
    def population_size(self):
        return sum([subpopulation.current_population_size for subpopulation in self.subpopulations])


class Subpopulation:
    def __init__(self,
                 manager,
                 params: dict,
                 discretization_damage: int,
                 discretization_volume: int):
        self.manager = manager
        self.params = params
        self.discretization_damage = discretization_damage
        self.discretization_volume = discretization_volume
        self.accept_step = True

    def initialize(self, population_size: int, nutrient: int, damage: int):
        pass

    def suggest_step(self):
        pass

    def step(self):
        pass

    @property
    def size(self):
        return None

    def add_cell(self, cell):
        pass

class DeterministicSubpopulation(Subpopulation):
    def __init__(self,
                 manager,
                 params: dict,
                 discretization_damage: int,
                 discretization_volume: int):
        super().__init__(manager=manager,
                         params=params,
                         discretization_damage=discretization_damage,
                         discretization_volume=discretization_volume)
        self.p = np.linspace(1, 2, self.discretization_volume)
        self.q = np.linspace(0, 1, self.discretization_damage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rhos = np.outer(1 / self.p, self.q)
            self.damage_death_rate = (self.rhos / (1 - self.rhos)) ** self.params["G"]
            self.damage_death_rate[np.isinf(self.damage_death_rate)] = self.damage_death_rate[
                ~np.isinf(self.damage_death_rate)].max()
        self.matrix = np.zeros((len(self.p), len(self.q)))
        self.suggested_matrix = np.zeros_like(self.matrix)
        self.consumed_nutrient_concentration = 0

    def initialize(self, population_size: int, nutrient: int, damage: int) -> None:
        mean_p, mean_q, var_p, var_q = \
            1 + np.random.random(), np.random.random(), np.random.random(), np.random.random()
        x, y = np.meshgrid(self.p, self.q)
        self.matrix = gaussian_2d(x.T, y.T, mean_p, mean_q, var_p, var_q)
        self.matrix = self.matrix / self.matrix.sum() * population_size

    def alarm_matrix(self, matrix: np.array) -> None:
        if (matrix < 0).sum() > 0:
            logging.debug(f"{matrix} failed the check (matrix)")
            self.accept_step = False

    def step(self) -> None:
        self.matrix = self.suggested_matrix
        # n_newborns = self.matrix[0, :].sum()
        # n_mutants = np.random.poisson(n_newborns*0.01*self.manager.simulation.delta_t)
        # for i in range(n_mutants):
        #     good = False
        #     while not good:
        #         index = np.random.choice(range(len(self.discretization_damage)), p=self.matrix[0, :]/self.matrix[0, :].sum())
        #         if self.matrix[0, index] >= 1:
        #             good = True
        #             self.matrix[0, index] -= 1



    def suggest_step(self) -> None:
        self.accept_step = True
        death_from = death(self.matrix, self.damage_death_rate, self.manager.simulation.chemostat_params["B"],
                           self.manager.simulation.delta_t)
        grow_from, grow_to = grow(self.matrix, self.manager.simulation.phi, self.params["A"],
                                  self.params["r"], self.params["E"],
                                  self.p, self.manager.simulation.delta_t, self.q)
        accumulate_from, accumulate_to = accumulate_damage(self.matrix, self.params["D"],
                                                           self.params["F"], self.manager.simulation.delta_t,
                                                           self.p, self.q)
        repair_from, repair_to = repair_damage(self.matrix, self.params["r"], self.manager.simulation.delta_t, self.p)
        new_matrix = self.matrix - death_from - grow_from + grow_to - accumulate_from + accumulate_to - repair_from + repair_to
        self.suggested_matrix = divide(new_matrix, self.q, self.params["a"])
        self.alarm_matrix(self.suggested_matrix)
        self.consumed_nutrient_concentration = \
            suggest_consumed_nutrient(self.matrix,
                                      self.manager.simulation.phi,
                                      self.params["C"],
                                      self.p,
                                      self.manager.simulation.delta_t)

    @property
    def size(self) -> float:
        return self.matrix.sum()

    def transform(self, subpopulation):
        self.matrix = np.zeros((self.discretization_volume, self.discretization_damage))
        for cell in subpopulation.cells:
            self.matrix[cell.nutrient - (self.discretization_volume - 1), cell.damage] += 1

    def __repr__(self):
        return f"Deterministic population, N={self.size}"

    def add_cell(self, cell):
        self.matrix[cell.nutrient, cell.damage] += 1


class StochasticSubpopulation(Subpopulation):
    def __init__(self,
                 manager,
                 params: dict,
                 discretization_damage: int,
                 discretization_volume: int):
        super().__init__(manager=manager,
                         params=params,
                         discretization_damage=discretization_damage,
                         discretization_volume=discretization_volume)

        self.critical_nutrient_amount = (discretization_volume - 1) * 2
        self.nutrient_to_volume_scaling_factor = 1 / (discretization_volume - 1)
        self.maximum_damage_amount = discretization_damage - 1

        self.cells = []
        self.consumed_nutrient_concentration = 0

    def initialize(self, population_size: int, nutrient: int, damage: int):
        if nutrient is None:
            nutrient = self.critical_nutrient_amount // 2
        if damage is None:
            damage = 0
        self.cells = [Cell(
            subpopulation=self,
            params=self.params,
            nutrient=nutrient,
            damage=damage) for _ in range(population_size)]

    def death_from_dilution(self, time_step_duration: float) -> None:
        expected_n_cells_to_remove = self.manager.simulation.chemostat_params["B"] * len(self.cells)
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove * time_step_duration, 1)[0]
        choose_from = list(filterfalse(lambda c: c.has_died, self.cells))
        dead_cells = np.random.choice(choose_from,
                                      size=min(len(choose_from), n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die(cause="dilution")

    def divide(self):
        new_cells = []
        for cell_obj in self.cells:
            offspring = cell_obj.reproduce()
            new_cells += offspring
        self.cells = new_cells

    def remove_dead_cells(self):
        self.cells = list(filterfalse(lambda c: c.has_died, self.cells))

    def step(self):
        for cell in self.cells:
            cell.live(self.manager.simulation.delta_t)
        self.death_from_dilution(self.manager.simulation.delta_t)
        self.divide()
        self.remove_dead_cells()

    def suggest_step(self):
        self.accept_step = True
        for cell in self.cells:
            cell.choose_nutrient_to_accumulate(self.manager.simulation.delta_t, self.manager.simulation.phi)
            cell.choose_damage_to_accumulate(self.manager.simulation.delta_t)
        suggested_damage_concentrations = [(max(cell.damage + cell.recently_accumulated_damage, 0) / self.maximum_damage_amount)/
                                           ((cell.nutrient + cell.recently_accumulated_nutrient) *
                                            self.nutrient_to_volume_scaling_factor)
                                           for cell in self.cells]
        self.alarm_damage_concentrations(suggested_damage_concentrations)
        self.alarm_damage_death_rate(suggested_damage_concentrations)
        if self.accept_step:
            self.consumed_nutrient_concentration = (self.params["C"] * self.manager.simulation.phi *
                                                    sum([cell.volume for cell in self.cells])) * \
                                                   self.manager.simulation.delta_t

    @property
    def size(self) -> int:
        return len(self.cells)

    def alarm_damage_concentrations(self, suggested_damage_concentrations: list):
        if any([dc > 1 for dc in suggested_damage_concentrations]):
            logging.debug(f"too high damage concentrations, rejecting step")
            self.accept_step = False

    def alarm_damage_death_rate(self, suggested_damage_concentrations: list):
        if any([dc != 1 and dc / (1 - dc) * self.manager.simulation.delta_t > 1
                for dc in suggested_damage_concentrations]):
            logging.debug(f"too high damage death rate, rejecting step")
            # logging.debug(str([dc / (1 - dc) * time_step_duration for dc in suggested_damage_concentrations]))
            self.accept_step = False

    def transform(self, subpopulation):
        self.cells = []
        for i in range(subpopulation.matrix.shape[0]):
            for j in range(subpopulation.matrix.shape[1]):
                self.cells.extend([Cell(subpopulation=self,
                                        params=self.params,
                                        nutrient=i + self.discretization_volume - 1,
                                        damage=j)
                                   for _ in range(round(subpopulation.matrix[i, j]))])

    def __repr__(self):
        return f"Stochastic population, N={self.size}"

    def add_cell(self, cell):
        self.cells.append(cell)


class Cell:
    # mutation rates
    asymmetry_mutation_rate = 0
    asymmetry_mutation_step = 0.01
    repair_mutation_rate = 0
    repair_mutation_step = 0.01

    def __init__(self,
                 subpopulation: StochasticSubpopulation,
                 params: dict,
                 nutrient: int,
                 damage=0):
        self.subpopulation = subpopulation
        self.params = params

        self.nutrient = nutrient
        self.damage = damage

        self._has_died = ""

        self.recently_accumulated_damage = 0
        self.recently_accumulated_nutrient = 0

    def choose_damage_to_accumulate(self, time_step_duration: float):
        expected_damage_to_accumulate = self.params["F"] * self.damage * time_step_duration + \
                                        self.params["D"] * (self.subpopulation.maximum_damage_amount+1) * self.volume
        prob_accumulation = expected_damage_to_accumulate * time_step_duration
        expected_damage_to_repair = (self.params["r"] * (self.subpopulation.maximum_damage_amount+1) * self.volume) * \
                                    (self.params["r"] > 0)
        prob_repair = (expected_damage_to_repair * time_step_duration) * (self.params["r"] > 0)
        self.recently_accumulated_damage = int(np.random.uniform(0, 1) < prob_accumulation) - \
                                           int(np.random.uniform(0, 1) < prob_repair)

    def choose_nutrient_to_accumulate(self, time_step_duration: float, nutrient_concentration: float) -> None:
        expected_nutrient = self.params["A"] * (1 - self.params["r"] / self.params["E"]) * \
                            self.volume * nutrient_concentration
        prob = expected_nutrient * time_step_duration  # poisson(0)
        self.recently_accumulated_nutrient = int(np.random.uniform(0, 1) < prob)

    def live(self, time_step_duration: float) -> None:
        self.nutrient += self.recently_accumulated_nutrient
        self.damage += self.recently_accumulated_damage

        # -rho/(1-rho) * n * delta_t
        working_concentration = self.damage_concentration if self.damage_concentration < 1 else \
            (self.subpopulation.maximum_damage_amount-1)/self.subpopulation.maximum_damage_amount / \
            ((self.subpopulation.critical_nutrient_amount/2)*self.subpopulation.nutrient_to_volume_scaling_factor)
        if np.random.uniform(0, 1) < time_step_duration * (
                working_concentration / (1 - working_concentration)) ** self.params["G"]:
            self.die(cause="damage")

    def reproduce(self) -> list:
        if self.volume >= self.critical_volume:
            offspring_dict = self.params.copy()
            offspring_asymmetry = self.params["a"]
            if np.random.uniform() < self.asymmetry_mutation_rate:
                offspring_asymmetry += np.random.choice([self.asymmetry_mutation_step, -self.asymmetry_mutation_step])
                offspring_asymmetry = max(min(1, offspring_asymmetry), 0)
            offspring_repair = self.params["r"]
            if np.random.uniform() < self.repair_mutation_rate:
                offspring_repair += np.random.choice([self.repair_mutation_step, -self.repair_mutation_step])
                offspring_repair = max(offspring_repair, 0)
            offspring_dict["a"] = offspring_asymmetry
            offspring_dict["r"] = offspring_repair
            offspring_dicts = [offspring_dict, self.params]
            if np.random.uniform(0, 1) < 0.5:
                offspring_dicts = offspring_dicts[::-1]
            damage1 = self.damage * (1 + self.params["a"]) // 2
            damage2 = self.damage - damage1
            cells = [Cell(subpopulation=self.subpopulation, params=offspring_dicts[0],
                        nutrient=self.nutrient//2, damage=damage1),
                    Cell(subpopulation=self.subpopulation, params=offspring_dicts[1],
                        nutrient=self.nutrient//2, damage=damage2)]
            res = []
            for cell in cells:
                if all([np.abs(cell.params[key] - self.params[key]) < 1e-5 for key in self.params.keys()]):
                    res.append(cell)
                else:
                    existing = False
                    for subpopulation in self.subpopulation.manager.simulation.subpopulations:
                        if all([cell.params[key] == subpopulation.params[key] for key in cell.params.keys()]):
                            subpopulation.add_cell(cell)
                            existing = True
                    if not existing:
                        new_subpopulation = SubpopulationManager(simulation=self.subpopulation.manager.simulation,
                                                                  params=cell.params,
                                                                  deterministic_threshold=self.subpopulation.manager.deterministic_threshold,
                                                                  starting_population_size=1,
                                                                  discretization_volume=self.subpopulation.discretization_volume,
                                                                  discretization_damage=self.subpopulation.discretization_damage,
                                                                  starting_nutrient=cell.nutrient,
                                                                  starting_damage=cell.damage)
                        self.subpopulation.manager.simulation.subpopulations.append(new_subpopulation)
        else:
            res = [self]
        res[0]._has_died = self.has_died
        return res

    @property
    def damage_concentration(self) -> float:
        return (self.damage / self.subpopulation.maximum_damage_amount) / self.volume

    @property
    def volume(self):
        return self.nutrient * self.subpopulation.nutrient_to_volume_scaling_factor

    @property
    def critical_volume(self):
        return self.subpopulation.critical_nutrient_amount * self.subpopulation.nutrient_to_volume_scaling_factor

    def die(self, cause: str) -> None:
        self._has_died = cause

    @property
    def damage(self) -> float:
        """
        :return: amount of somatic damage accumulated by the cell
        """
        return self._damage

    @damage.setter
    def damage(self, damage):
        self._damage = min(self.subpopulation.maximum_damage_amount, max(damage, 0))

    @property
    def nutrient(self) -> int:
        return self._nutrient

    @nutrient.setter
    def nutrient(self, nutrient):
        self._nutrient = nutrient

    @property
    def has_died(self) -> str:
        """
        :return: if cell has died at the current time step
        """
        return self._has_died

    def __repr__(self):
        return f"Damage: {self.damage}, Alive: {not self.has_died}"


class SubpopulationManager:
    def __init__(self,
                 simulation: Simulation,
                 params: dict,
                 deterministic_threshold: int,
                 starting_population_size: int,
                 discretization_volume: int,
                 discretization_damage: int,
                 starting_nutrient=None,
                 starting_damage=None):
        self.simulation = simulation
        self.deterministic_threshold = deterministic_threshold
        self.params = params
        self.deterministic_simulation = DeterministicSubpopulation(manager=self,
                                                                   params=params,
                                                                   discretization_damage=discretization_damage,
                                                                   discretization_volume=discretization_volume)
        self.stochastic_simulation = StochasticSubpopulation(manager=self,
                                                             params=params,
                                                             discretization_damage=discretization_damage,
                                                             discretization_volume=discretization_volume)
        self.current_subpopulation = self.stochastic_simulation \
            if starting_population_size < self.deterministic_threshold else self.deterministic_simulation
        self.waiting_subpopulation = self.deterministic_simulation \
            if self.current_subpopulation is self.stochastic_simulation else self.stochastic_simulation

        self.current_subpopulation.initialize(starting_population_size, starting_nutrient, starting_damage)

    def suggest_step(self):
        self.current_subpopulation.suggest_step()

    def step(self):
        self.current_subpopulation.step()
        if (isinstance(self.current_subpopulation, StochasticSubpopulation)
            and self.current_population_size > self.deterministic_threshold) \
                or (isinstance(self.current_subpopulation, DeterministicSubpopulation)
                    and self.current_population_size < self.deterministic_threshold):
            self.current_subpopulation, self.waiting_subpopulation = self.waiting_subpopulation, \
                self.current_subpopulation
            self.current_subpopulation.transform(self.waiting_subpopulation)

    @property
    def consumed_nutrient_concentration(self):
        return self.current_subpopulation.consumed_nutrient_concentration

    @property
    def current_population_size(self):
        return self.current_subpopulation.size

    def add_cell(self, cell: Cell):
        self.current_subpopulation.add_cell(cell)


class History:
    def __init__(self, simulation_obj: Simulation, save_path: str):
        self.simulation = simulation_obj
        self.population_sizes = [[] for subpopulation in self.simulation.subpopulations]
        self.times = []
        self.save_path = save_path
        Path(self.save_path).mkdir(exist_ok=True)
        self.starting_time = tm.time()

    def record(self) -> None:
        for i, subpopulation in enumerate(self.simulation.subpopulations):
            self.population_sizes[i].append(subpopulation.current_population_size)
        self.times.append(self.simulation.time)

    def save(self):
        with open(f"{self.save_path}/population_size_estimate.txt", "w") as fl:
            fl.write(",".join(map(str, self.times)) + '\n')
            for population_history in self.population_sizes:
                fl.write(",".join(map(str, population_history)) + '\n')

