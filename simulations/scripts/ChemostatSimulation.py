# Test commit

import pathlib
import random
import string
import time

from wonderwords import RandomWord

import pandas as pd
import datetime
from pathlib import Path
import numpy as np
import os
import json
import argparse
import sqlite3
import atexit
import logging

from itertools import filterfalse

PRECISION = 3
logging.basicConfig(level=logging.INFO)


def calculate_rho(B, D, F, r):
    B_prime = B - F
    D_prime = max(D-r, 0)
    first_term = (B_prime + D_prime)/(1-B_prime)
    second_term = first_term**2
    third_term = 4 * D_prime/(1-B_prime)
    term23 = np.sqrt(second_term + third_term)
    return 0.5*(-first_term + term23), 0.5*(-first_term - term23)


def calculate_phi(A, B, D, E, F, r):
    D_prime = max(D-r, 0)
    A_prime = A*(1-r/E)
    rho = calculate_rho(B, D, F, r)
    if rho[0] == 0:
        return B/A_prime, B/A_prime
    else:
        return D_prime/(A_prime*rho[0]) + F/A_prime, D_prime/(A_prime*rho[1] + F/A_prime)


def calculate_n(A, B, C, D, E, F, r):
    phi = calculate_phi(A, B, D, E, F, r)
    return (B/C)*(1-phi[0])/phi[0], (B/C)*(1-phi[1])/phi[1]


class Chemostat:
    def __init__(self,
                 volume_val: float,
                 dilution_rate: float,
                 medium_richness: float = 1.0,
                 n_cells=None,
                 n_array_discretization_steps: int = 1001,
                 starting_nutrient_concentration: float = 1,
                 asymmetry=0.0,
                 damage_repair_intensity=0.0):
        self.V = volume_val
        self.D = dilution_rate
        self.medium_richness = medium_richness
        self.nutrient_concentration = starting_nutrient_concentration
        self._cells = []
        self._n = 0
        if n_cells:
            self.populate_with_cells(n_cells, asymmetry=asymmetry, damage_repair_intensity=damage_repair_intensity)
        self.n_array_bins = np.linspace(0, 1, n_array_discretization_steps)
        self._n_array = None

    @property
    def N(self) -> int:
        return self._n

    @property
    def cells(self) -> list:
        return self._cells

    def populate_with_cells(self, n_cells: int, asymmetry: float, damage_repair_intensity: float) -> None:
        self._cells += [Cell(chemostat=self,
                             cell_id=i,
                             asymmetry=asymmetry,
                             damage_repair_intensity=damage_repair_intensity) for i in range(n_cells)]
        self._n = len(self._cells)

    def cells_from_n_array(self):
        if self.n_array is not None:
            self.cells = []
            index = 0
            for damage_concentration, n_cells in zip(self.n_array_bins, self.n_array):
                final_index = index + n_cells.round().astype(int)
                self.cells += [Cell(chemostat=self,
                                    cell_id=i,
                                    damage=damage_concentration * Cell.critical_volume,
                                    asymmetry=0,
                                    damage_repair_intensity=0) for i in range(index, final_index)]
                index = final_index
            self.n_array = None

    def dilute(self, time_step_duration: float) -> None:
        expected_n_cells_to_remove = self.D * self.N / self.V
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove * time_step_duration, 1)[0]
        dead_cells = np.random.choice(list(filterfalse(lambda c: c.has_died, self.cells)),
                                      size=min(self.N, n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die(cause="dilution")
        self._n = len(list(filterfalse(lambda c: c.has_died, self.cells)))
        self.nutrient_concentration += (self.D / self.V) * (1 - self.nutrient_concentration) * time_step_duration

    @cells.setter
    def cells(self, cells):
        self._cells = cells
        self._n = len(self._cells)

    def set_n_array(self):
        if self._n_array is None:
            self._n_array = np.zeros_like(self.n_array_bins)
            damage_step = 1 / (len(self.n_array_bins) - 1)
            np.add.at(self._n_array,
                      np.digitize([cell.damage_concentration for cell in self.cells],
                                  self.n_array_bins + damage_step / 2),
                      1)

    @property
    def n_array(self):
        return self._n_array

    @n_array.setter
    def n_array(self, n_array: np.array):
        self._n_array = n_array
        if n_array is not None:
            self._n = n_array.sum()


class Cell:
    # Nutrient accumulation
    critical_nutrient_amount = 10
    nutrient_accumulation_rate = 1
    nutrient_to_volume_scaling_factor = 1
    critical_volume = critical_nutrient_amount * nutrient_to_volume_scaling_factor

    # Damage accumulation
    damage_accumulation_exponential_component = 0
    damage_accumulation_linear_component = 0
    damage_survival_dependency = 1
    damage_reproduction_dependency = False

    # mutation rates
    asymmetry_mutation_rate = 0
    asymmetry_mutation_step = 0.01
    repair_mutation_rate = 0
    repair_mutation_step = 0.01

    # lambda_large_lookup
    lambda_large_lookup_path = f"../tables/lambda_large_lookup_table.csv"
    lambda_large_lookup = None

    # damage repair
    repair_cost_coefficient = 1

    def __init__(self,
                 chemostat: Chemostat,
                 cell_id: int,
                 parent_id=None,
                 asymmetry=0.0,
                 age=0,
                 damage=0.0, damage_repair_intensity=0.0):
        self.chemostat = chemostat
        self._id = cell_id
        self._parent_id = parent_id
        self._age = age
        self._damage = damage
        self.starting_damage = damage
        self.asymmetry = asymmetry
        self._has_reproduced = False
        self._has_died = ""
        self.damage_repair_intensity = damage_repair_intensity
        self.recently_accumulated_damage = 0
        self.nutrient = self.critical_nutrient_amount
        self.recently_accumulated_nutrient = 0

    def choose_damage_to_accumulate(self, time_step_duration):
        self.recently_accumulated_damage = (Cell.damage_accumulation_exponential_component * self._damage
                                            + Cell.damage_accumulation_linear_component -
                                            self.damage_repair_intensity) * self.volume \
                                           * time_step_duration

    def choose_nutrient_to_accumulate(self, time_step_duration: float) -> None:
        expected_nutrient = self.volume * \
                            self.chemostat.nutrient_concentration * \
                            self.nutrient_accumulation_rate * self.chemostat.medium_richness * \
                            (1-self.damage_repair_intensity/self.repair_cost_coefficient) * time_step_duration
        self.recently_accumulated_nutrient = expected_nutrient
        # self.recently_accumulated_nutrient = np.random.poisson(expected_nutrient, 1)[0]

    def live(self, time_step_duration: float) -> None:
        self._age += time_step_duration
        self.nutrient += self.recently_accumulated_nutrient
        self._damage += self.recently_accumulated_damage

        # -rho/(1-rho) * n * delta_t
        if np.random.uniform(0, 1) < time_step_duration * (
                self.damage_concentration / (1 - self.damage_concentration)) ** Cell.damage_survival_dependency:
            self.die(cause="damage")

    def reproduce(self, offspring_id: int) -> list:
        self._has_reproduced = self.volume >= self.critical_volume * 2
        if self.has_reproduced:
            offspring_asymmetry = self.asymmetry
            if np.random.uniform() < self.asymmetry_mutation_rate:
                offspring_asymmetry += np.random.choice([self.asymmetry_mutation_step, -self.asymmetry_mutation_step])
            offspring_damage_repair_intensity = self.damage_repair_intensity
            if np.random.uniform() < self.repair_mutation_rate:
                offspring_damage_repair_intensity += np.random.choice([self.repair_mutation_step,
                                                                       -self.repair_mutation_step])
            res = [type(self)(chemostat=self.chemostat,
                              cell_id=offspring_id,
                              parent_id=self.id,
                              asymmetry=offspring_asymmetry,
                              damage=self.damage * (1 + self.asymmetry) / 2,
                              damage_repair_intensity=offspring_damage_repair_intensity
                              ),
                   type(self)(chemostat=self.chemostat,
                              cell_id=offspring_id + 1,
                              parent_id=self.id,
                              asymmetry=offspring_asymmetry,
                              damage=self.damage * (1 - self.asymmetry) / 2,
                              damage_repair_intensity=offspring_damage_repair_intensity
                              ),
                   ]
        else:
            res = [self]
        return res

    @property
    def damage_concentration(self) -> float:
        return self.damage / self.volume

    @property
    def volume(self):
        return self.nutrient * self.nutrient_to_volume_scaling_factor

    def die(self, cause: str) -> None:
        self._has_died = cause

    @property
    def id(self) -> int:
        """
        :return: a unique id of a cell
        """
        return self._id

    @property
    def parent_id(self) -> int:
        """
        :return: id of the parent cell
        """
        return self._parent_id

    @property
    def age(self) -> int:
        """
        :return: number of time steps passed since the timestep of cell's birth
        """
        return self._age

    @property
    def asymmetry(self) -> float:
        """
        :return: the asymmetry value of a cell. Impacts the damage inheritance of the daughter cells.
        The inherited damage for daughter cells is calculated as damage*(1+asymmetry)/2 and damage*(1-asymmetry)/2
        """
        return self._asymmetry

    @property
    def damage(self) -> float:
        """
        :return: amount of somatic damage accumulated by the cell
        """
        return self._damage

    @property
    def damage_repair_intensity(self) -> float:
        return self._damage_repair_intensity

    @property
    def has_reproduced(self) -> bool:
        """
        :return: if cell has reproduced at the current time step
        """
        return self._has_reproduced

    @property
    def has_died(self) -> str:
        """
        :return: if cell has died at the current time step
        """
        return self._has_died

    @property
    def starting_damage(self) -> float:
        """

        :return: amount of damage cell got at birth
        """
        return self._starting_damage

    @starting_damage.setter
    def starting_damage(self, damage):
        self._starting_damage = damage

    @damage.setter
    def damage(self, damage):
        self._damage = max(damage, 0)

    @asymmetry.setter
    def asymmetry(self, asymmetry):
        self._asymmetry = min(max(asymmetry, 0), 1)

    @damage_repair_intensity.setter
    def damage_repair_intensity(self, damage_repair_intensity):
        self._damage_repair_intensity = min(1, max(damage_repair_intensity, 0))

    def __repr__(self):
        return f"ID: {self.id}, Age: {self.age}, Damage: {self.damage}, " \
               f"Alive: {not self.has_died}, Reproduced: {self.has_reproduced}"


class Simulation:
    def __init__(self,
                 parameters: dict,
                 n_starting_cells: int,
                 save_path: str,
                 mode: str,
                 n_threads: int, n_procs: int,
                 deterministic_threshold: int,
                 smart_initialization: bool,
                 run_name="",
                 write_cells_table=False,
                 add_runs="",
                 record_history=True):
        Cell.damage_repair_mode = parameters["cell_parameters"]["damage_repair_mode"]
        Cell.damage_accumulation_exponential_component = \
            parameters["cell_parameters"]["damage_accumulation_exponential_component"]
        Cell.damage_accumulation_linear_component = \
            parameters["cell_parameters"]["damage_accumulation_linear_component"]
        Cell.damage_survival_dependency = parameters["cell_parameters"]["damage_survival_dependency"]
        Cell.damage_reproduction_dependency = parameters["cell_parameters"]["damage_reproduction_dependency"]

        Cell.nutrient_accumulation_rate = parameters["cell_parameters"]["nutrient_accumulation_rate"]
        Cell.critical_nutrient_amount = parameters["cell_parameters"]["critical_nutrient_amount"]

        Cell.asymmetry_mutation_step = parameters["cell_parameters"]["asymmetry_mutation_step"]
        Cell.asymmetry_mutation_rate = parameters["cell_parameters"]["asymmetry_mutation_rate"]
        Cell.repair_mutation_step = parameters["cell_parameters"]["repair_mutation_step"]
        Cell.repair_mutation_rate = parameters["cell_parameters"]["repair_mutation_rate"]
        Cell.repair_cost_coefficient = parameters["cell_parameters"]["max_repair"]
        Cell.nutrient_to_volume_scaling_factor = parameters["cell_parameters"]["nutrient_to_volume_scaling_factor"]
        if add_runs:
            run_id = add_runs.rstrip("/").split("/")[-1].split("_")[0]
            max_existing_thread = max([int(file.stem.split("_")[-1]) for file in Path(add_runs).glob("*.sqlite")])
        else:
            run_id = str(round(datetime.datetime.now().timestamp() * 1000000))
            max_existing_thread = 0
        if not run_name:
            letter = np.random.choice(list(string.ascii_lowercase))
            run_name = "_".join([RandomWord().word(starts_with=letter, include_parts_of_speech=["adjective"]),
                                 RandomWord().word(starts_with=letter, include_parts_of_speech=["noun"])])
        run_name = "_" + run_name
        (Path(save_path) / Path(f"{run_id}{run_name}")).mkdir(exist_ok=True)

        core_params = dict(
            volume=parameters["chemostat_parameters"]["volume"],
            dilution_rate=parameters["chemostat_parameters"]["dilution_rate"],
            nutrient_accumulation_rate=parameters["cell_parameters"]["nutrient_accumulation_rate"],
            nutrient_to_volume_scaling_factor=1,
            medium_richness=parameters["medium_richness"],
            critical_nutrient_amount=parameters["cell_parameters"]["critical_nutrient_amount"],
            damage_accumulation_linear_component=parameters["cell_parameters"]["damage_accumulation_linear_component"],
            damage_accumulation_exponential_component=parameters["cell_parameters"]["damage_accumulation_exponential_component"],
            max_repair=parameters["cell_parameters"]["max_repair"]
        )
        time_step_params = dict(starting_time_step_duration=0.0001, delta_time_step=0.1)
        changing_environment_params = dict(changing_environment_prob=parameters["changing_environment_probability"],
                                           harsh_environment_frac=parameters["harsh_environment_frac"])
        history_params_list = [dict(run_id=run_id,
                                    run_name=run_name,
                                    thread_id=i + 1,
                                    save_path=save_path,
                                    write_cells_table=write_cells_table,
                                    record_history=record_history) for i in
                               range(max_existing_thread, max_existing_thread + n_threads)]
        self.threads = [SimulationManager(core_params=core_params,
                                          starting_population_size=n_starting_cells,
                                          starting_asymmetry=parameters["asymmetry"],
                                          starting_damage_repair_intensity=parameters["damage_repair_intensity"],
                                          discretization_n_steps=1001,
                                          deterministic_threshold=deterministic_threshold,
                                          mode=mode,
                                          time_step_params=time_step_params,
                                          history_params=history_params,
                                          changing_environment_params=changing_environment_params,
                                          smart_initialization=smart_initialization)
                        for history_params in history_params_list]
        self.n_procs = n_procs if mode in ["local", "interactive"] else 1

        # Write parameters needed to identify simulation
        with open(f"{save_path}/{run_id}{run_name}/params.json", "w") as fl:
            json.dump(parameters, fl)

    def run_thread(self, thread_number: int, n_steps: int) -> None:
        self.threads[thread_number].run(n_steps=n_steps)

    def run(self, n_steps: int) -> None:
        if self.n_procs == 1:
            for thread in self.threads:
                thread.run(n_steps=n_steps)
        elif self.n_procs > 1:
            for i in tqdm(range(0, len(self.threads), self.n_procs)):
                processes = []
                for j in range(i, min(i + self.n_procs, len(self.threads))):
                    processes.append(multiprocessing.Process(target=self.run_thread,
                                                             args=(j, n_steps)))
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()


class SimulationManager:
    def __init__(self,
                 core_params: dict,
                 starting_population_size: int,
                 starting_asymmetry: float,
                 starting_damage_repair_intensity: float,
                 discretization_n_steps: int,
                 deterministic_threshold: int,
                 mode: str,
                 time_step_params: dict,
                 history_params: dict,
                 smart_initialization: bool,
                 changing_environment_params: dict = None,
                 ):
        self.smart_initialization = smart_initialization
        self.core_params = core_params

        self.discrete_simulation = DiscreteSimulation(manager=self,
                                                      chemostat_obj=Chemostat(
                                                          volume_val=self.core_params["volume"],
                                                          dilution_rate=self.core_params["dilution_rate"],
                                                          medium_richness=self.core_params["medium_richness"],
                                                          n_cells=starting_population_size,
                                                          asymmetry=starting_asymmetry,
                                                          damage_repair_intensity=starting_damage_repair_intensity))

        n_array = np.zeros(discretization_n_steps)
        n_array[0] = starting_population_size
        print(self.core_params["nutrient_accumulation_rate"], self.core_params["nutrient_to_volume_scaling_factor"],
              self.core_params["critical_nutrient_amount"], self.core_params["volume"])
        self.continuous_simulation = ContinuousSimulation(manager=self,
                                                          n_array=n_array,
                                                          constants=dict(
                                                              cell_growth_rate=self.core_params[
                                                                                   "nutrient_accumulation_rate"] *
                                                                               self.core_params[
                                                                                   "nutrient_to_volume_scaling_factor"] *
                                                                               self.core_params["medium_richness"],
                                                              dilution_rate=self.core_params["dilution_rate"] /
                                                                            self.core_params["volume"],
                                                              nutrient_acquisition_rate=
                                                              self.core_params["nutrient_accumulation_rate"] *
                                                              self.core_params["nutrient_to_volume_scaling_factor"] *
                                                              self.core_params["critical_nutrient_amount"] /
                                                              self.core_params["volume"],
                                                              damage_accumulation_linear_component=self.core_params[
                                                                  "damage_accumulation_linear_component"],
                                                              damage_accumulation_exponential_component=self.core_params["damage_accumulation_exponential_component"],
                                                              max_repair=self.core_params["max_repair"]
                                                          ),
                                                          asymmetry=starting_asymmetry,
                                                          repair=starting_damage_repair_intensity)

        self.deterministic_threshold = deterministic_threshold
        self.current_population_size = starting_population_size
        self.current_simulation = self.discrete_simulation \
            if self.current_population_size < self.deterministic_threshold else self.continuous_simulation
        self.waiting_simulation = self.discrete_simulation if self.current_simulation is self.continuous_simulation \
            else self.continuous_simulation

        # Time steps
        self.time_step_duration = time_step_params["starting_time_step_duration"]
        self.delta_time_step = time_step_params["delta_time_step"]

        # Drawing
        self.mode = mode
        if self.mode == "interactive":
            self.drawer = Drawer(self)

        # History
        self.history = History(self, **history_params) if history_params["record_history"] else None

        # Changing environment parameters
        if changing_environment_params is None:
            changing_environment_params = {"changing_environment_prob": 0, "harsh_environment_frac": 1}

        if changing_environment_params["changing_environment_prob"] == 0:
            self.environment_switch_probs = [0, 0]
            if changing_environment_params["harsh_environment_frac"] == 1:
                self.current_environment = True
            elif changing_environment_params["harsh_environment_frac"] == 0:
                self.current_environment = False
            else:
                raise ValueError(f"If changing_environment_prob == 0, "
                                 f"harsh_environment_frac should be 0 or 1, not "
                                 f"{changing_environment_params['harsh_environment_frac']}")
        else:
            if changing_environment_params["harsh_environment_frac"] in [0, 1]:
                raise ValueError(f"If harsh_environment_frac == "
                                 f"{changing_environment_params['harsh_environment_frac']}, "
                                 f"changing_environment_prob should be 0")
            self.environment_switch_probs = [changing_environment_params["changing_environment_prob"] /
                                             (2 * (1 - changing_environment_params["harsh_environment_frac"])),
                                             changing_environment_params["changing_environment_prob"] /
                                             (2 * changing_environment_params["harsh_environment_frac"])]
            self.current_environment = True
        self.changing_environment_val = {True: self.core_params["damage_accumulation_linear_component"],
                                         False: 0}

        self.time = 0

    def _change_environment(self) -> None:
        if self.environment_switch_probs[int(self.current_environment)] \
                and random.uniform(0, 1) < self.environment_switch_probs[int(self.current_environment)] * \
                self.time_step_duration:
            self.current_environment = not self.current_environment
            self.continuous_simulation.constants["damage_accumulation_linear_component"] = \
                self.changing_environment_val[self.current_environment]
            Cell.damage_accumulation_linear_component = self.changing_environment_val[self.current_environment]

    def step(self, step_number: int):
        # Switch environment
        self._change_environment()

        # Do step in continuous or discrete simulation
        self.time_step_duration = self.current_simulation.step(self.time_step_duration, self.delta_time_step)
        self.time += self.time_step_duration

        # If deterministic threshold has been crossed, swap current and waiting simulations
        if (self.current_population_size > self.deterministic_threshold) != \
                (self.current_simulation.population_size > self.deterministic_threshold):
            self.current_simulation, self.waiting_simulation = self.waiting_simulation, self.current_simulation
            self.current_simulation.transform(self.waiting_simulation)

        self.current_population_size = self.current_simulation.population_size

        # Record history
        if self.history is not None:
            self.history.record(step_number)

    def run(self, n_steps: int) -> None:
        np.random.seed((os.getpid() * int(datetime.datetime.now().timestamp()) % 123456789))
        if self.mode in ["local", "interactive"]:
            iterator = tqdm(range(n_steps))
        else:
            iterator = range(n_steps)

        tt, nn = [], []
        start_time = time.time()
        for step_number in iterator:
            self.step(step_number)
            self.time_step_duration = min(self.time_step_duration, 0.00001)
            tt.append(self.time)
            nn.append(self.current_population_size)
            dif = None
            for i, t in enumerate(tt[::-1]):
                if tt[-1] - tt[-(i+1)] > 20:
                    dif = np.array(nn[-(i+1):]).mean().round().astype(int) == int(round(nn[-1])) and \
                          (max(nn[-(i+1):]) - min(nn[-(i+1):]))/nn[-1] < 0.001
                    dif = dif or all([el < 1 for el in nn])
                    tt = tt[-(i + 1):]
                    nn = nn[-(i + 1):]
                    break
            if step_number % 1000 == 0:
                print(self.current_population_size, self.current_simulation.phi, self.time)
            overtime = time.time() - start_time > 60*10
            if dif or overtime:
                Path("equilibria/").mkdir(exist_ok=True)
                filename = "_".join(list(map(str, [
                    self.continuous_simulation.asymmetry,
                    self.continuous_simulation.repair,
                    self.continuous_simulation.constants["cell_growth_rate"],
                    self.continuous_simulation.constants["dilution_rate"],
                    self.continuous_simulation.constants["nutrient_acquisition_rate"],
                    self.continuous_simulation.constants["damage_accumulation_linear_component"],
                    self.continuous_simulation.constants["damage_accumulation_exponential_component"],
                    self.continuous_simulation.constants["max_repair"]
                ])))
                with open(f"equilibria/{filename}.txt", "w") as fl:
                    fl.write(f"{self.continuous_simulation.phi}\n")
                    fl.write(" ".join(list(map(str, self.continuous_simulation.n_array))) + "\n")
                    fl.write(f"{self.current_population_size}\n")
                    if overtime:
                        fl.write("overtime\n")
                    elif all([el < 1 for el in nn]):
                        fl.write("probably dying out")
                    print("equilibrium population size: ", self.current_population_size)
                break
            if self.mode == "interactive":
                self.drawer.draw_step(step_number, self.time_step_duration)
            if self.current_population_size == 0:
                logging.info("The population died out.")
                break
        if self.history:
            self.history.save()
            self.history.SQLdb.close()


class DiscreteSimulation:
    def __init__(self, manager: SimulationManager, chemostat_obj: Chemostat):
        self.manager = manager
        self.chemostat = chemostat_obj

    def step(self, time_step_duration: float, delta_time_step: float) -> float:
        accept_step = False
        increase_time_step = random.uniform(0, 1) < 0.01
        while not accept_step:
            accept_step = True
            for cell in self.chemostat.cells:
                cell.choose_nutrient_to_accumulate(time_step_duration)
                cell.choose_damage_to_accumulate(time_step_duration)
            suggested_damage_concentrations = [(cell.damage + cell.recently_accumulated_damage) /
                                               ((cell.nutrient + cell.recently_accumulated_nutrient) *
                                                cell.nutrient_to_volume_scaling_factor)
                                               for cell in self.chemostat.cells]

            if any([dc > 1 for dc in suggested_damage_concentrations]):
                print(1)
            if any([dc / (1 - dc) * time_step_duration > 1 for dc in suggested_damage_concentrations]):
                print([dc / (1 - dc) * time_step_duration for dc in suggested_damage_concentrations])
            if sum([cell.recently_accumulated_nutrient for cell in self.chemostat.cells]) / self.chemostat.V > \
                    self.chemostat.nutrient_concentration:
                print(3, self.chemostat.N, time_step_duration)
            if any([dc > 1 for dc in suggested_damage_concentrations]) or \
                    any([dc / (1 - dc) * time_step_duration > 1 for dc in suggested_damage_concentrations]) or \
                    sum([cell.recently_accumulated_nutrient for cell in self.chemostat.cells]) / self.chemostat.V > \
                    self.chemostat.nutrient_concentration:
                accept_step = False
                increase_time_step = False
                time_step_duration -= time_step_duration * delta_time_step

        # Time passes
        for cell in self.chemostat.cells:
            cell.live(time_step_duration)
        self.chemostat.nutrient_concentration -= sum([cell.recently_accumulated_nutrient
                                                      for cell in self.chemostat.cells]) / self.chemostat.V

        # Cells are diluted
        self.chemostat.dilute(time_step_duration)

        # Alive cells reproduce
        new_cells = []
        for cell in filterfalse(lambda cell_obj: cell_obj.has_died, self.chemostat.cells):
            offspring_id = max([cell.id for cell in self.chemostat.cells] + [cell.id for cell in new_cells]) + 1
            new_cells += cell.reproduce(offspring_id)

        # Move to the next time step
        self.chemostat.cells = new_cells
        time_step_duration += time_step_duration * delta_time_step * int(increase_time_step)
        return time_step_duration

    def transform(self, continuous_simulation):
        self.chemostat.nutrient_concentration = continuous_simulation.phi
        self.chemostat.cells = []
        index = 0
        for damage_concentration, n_cells in zip(continuous_simulation.n_array_bins,
                                                 continuous_simulation.n_array):
            final_index = index + n_cells.round().astype(int)
            self.chemostat.cells += [Cell(chemostat=self.chemostat,
                                          cell_id=i,
                                          damage=damage_concentration * Cell.critical_volume,
                                          asymmetry=0,
                                          damage_repair_intensity=0) for i in range(index, final_index)]
            index = final_index

    @property
    def population_size(self):
        return self.chemostat.N

    @property
    def mean_damage_concentration(self):
        return sum([cell.damage_concentration for cell in self.chemostat.cells])/self.chemostat.N


class ContinuousSimulation:
    def __init__(self, manager: SimulationManager,
                 constants: dict,
                 asymmetry: float,
                 repair: float,
                 n_array: np.array,
                 phi: float = 1.0):
        self.manager = manager
        self.constants = constants
        self.asymmetry = asymmetry
        self.repair = repair
        self.phi = phi
        self.n_array = n_array

        print(constants)
        if self.manager.smart_initialization and self.repair != self.constants["max_repair"] and Path("equilibria").exists() and \
                len(list(Path("equilibria").glob("*"))) > 0:
            # closest_equilibrium = self.find_closest_equilibrium()
            closest_equilibrium = None
            if closest_equilibrium is not None:
                print(f"Initializing with {closest_equilibrium}")
                with closest_equilibrium.open("r") as fl:
                    self.phi = float(fl.readline().strip())
                    self.n_array = np.array(list(map(float, fl.readline().strip().split())))
            else:
                try:
                    rho = calculate_rho(B=self.constants["dilution_rate"],
                                        D=self.constants["damage_accumulation_linear_component"],
                                        F=self.constants["damage_accumulation_exponential_component"],
                                        r=self.repair)
                    phi = calculate_phi(A=self.constants["cell_growth_rate"],
                                        B=self.constants["dilution_rate"],
                                        D=self.constants["damage_accumulation_linear_component"],
                                        E=self.constants["max_repair"],
                                        F=self.constants["damage_accumulation_exponential_component"],
                                        r=self.repair)
                    n = calculate_n(A=self.constants["cell_growth_rate"],
                                    B=self.constants["dilution_rate"],
                                    C=self.constants["nutrient_acquisition_rate"],
                                    D=self.constants["damage_accumulation_linear_component"],
                                    E=self.constants["max_repair"],
                                    F=self.constants["damage_accumulation_exponential_component"],
                                    r=self.repair)
                    print(rho, phi, n)
                    rho = rho[0]
                    phi = phi[0]
                    n = n[0]
                    if all([el > 0 for el in [rho, phi, n]]):
                        self.phi = phi
                        for i, j in zip(range(1002), np.linspace(0, 1, 1001)):
                            if round(rho, 3) == round(j, 3):
                                self.n_array = np.zeros(1001)
                                self.n_array[i] = n
                                break
                        print(f"nutrient: ", self.phi)
                        # print(f"population: ", list(self.n_array))
                        print(f"population_size: ", self.n_array.sum())
                except ZeroDivisionError:
                    self.phi = phi
                    self.n_array = n_array
        print("initialized", self.phi, self.n_array.sum())
        self.n_array_bins = np.linspace(0, 1, len(self.n_array))

    def find_closest_equilibrium(self) -> pathlib.Path:
        """
        Finds the file in equilibria folder that has the parameters most resembling those of the current simulation.
        :return:
        """
        equilibrium_files = list(Path("equilibria").glob("*"))
        existing_equilibria = list(map(lambda el: list(map(float, str(el.stem).split("_"))),
                                       equilibrium_files))
        my_params = np.array([
            self.asymmetry,
            self.repair,
            self.constants["cell_growth_rate"],
            self.constants["dilution_rate"],
            self.constants["nutrient_acquisition_rate"],
            self.constants["damage_accumulation_linear_component"],
            self.constants["damage_accumulation_exponential_component"],
            self.constants["max_repair"]])
        closest_equilibria = list(filter(lambda eq: ((np.array(eq) - my_params) != 0).sum() <= 1, existing_equilibria))
        if not closest_equilibria:
            return None
        else:
            closest_equilibrium_file = equilibrium_files[np.argmin([sum(np.abs((np.array(eq) - my_params))) +
                                                                    1e100*int(eq not in closest_equilibria)
                                                                    for eq in existing_equilibria])]
            return closest_equilibrium_file

    def step(self, time_step_duration: float, delta_time_step: float):
        accept_step = False
        increase_time_step = random.uniform(0, 1) < 0.01
        while not accept_step:
            proposed_new_phi, accept_step = self.propose_new_phi(delta_t=time_step_duration)
            n_array, accept_step = self.die(delta_t=time_step_duration, run=accept_step)

            # TODO: do it separately for different asymmetries
            n_array, accept_step = self.reproduce(n_array=n_array, phi=proposed_new_phi,
                                                  delta_t=time_step_duration, run=accept_step)

            n_array, accept_step = self.accumulate_damage(n_array=n_array, delta_t=time_step_duration, run=accept_step)

            if accept_step:
                self.n_array = n_array
                self.phi = proposed_new_phi
            else:
                time_step_duration -= time_step_duration * delta_time_step
                increase_time_step = False
        time_step_duration += time_step_duration * delta_time_step * int(increase_time_step)
        return time_step_duration

    def transform(self, discrete_simulation: DiscreteSimulation):
        self.phi = discrete_simulation.chemostat.nutrient_concentration
        self.n_array = np.zeros_like(self.n_array_bins)
        damage_step = 1 / (len(self.n_array_bins) - 1)
        np.add.at(self.n_array,
                  np.digitize([cell.damage_concentration for cell in discrete_simulation.chemostat.cells],
                              self.n_array_bins + damage_step / 2),
                  1)

    @property
    def population_size(self):
        return self.n_array.sum()

    def propose_new_phi(self, delta_t: float) -> (float, bool):
        accept_step = True
        proposed_new_phi = self.phi + ContinuousSimulation.derivative_phi(self.n_array.sum(),
                                                                          self.phi,
                                                                          self.constants["dilution_rate"],
                                                                          self.constants["nutrient_acquisition_rate"]) * delta_t
        if proposed_new_phi < 0:
            accept_step = False
        return proposed_new_phi, accept_step

    def _death_func(self) -> float:
        return np.divide(self.n_array * self.n_array_bins, 1 - self.n_array_bins,
                         out=self.n_array * (1 - self.constants["dilution_rate"]),
                         where=self.n_array_bins != 1) + self.constants["dilution_rate"] * self.n_array

    def die(self, delta_t: float, run: bool) -> (np.array, bool):
        if not run:
            return None, False
        accept_step = True
        dead = self._death_func() * delta_t
        if (dead > self.n_array).sum() > 0:
            accept_step = False
        return self.n_array - dead, accept_step

    def accumulate_damage(self, n_array: np.array, delta_t: float, run: bool) -> (np.array, bool):
        if not run:
            return None, False
        new_n_array = np.zeros_like(n_array)
        damage_step = 1 / (len(self.n_array_bins) - 1)
        increment = ContinuousSimulation.derivative_rho(rho_vector=self.n_array_bins,
                                                        current_phi=self.phi,
                                                        damage_accumulation_linear_component=max(self.constants["damage_accumulation_linear_component"] - self.repair, 0),
                                                        damage_accumulation_exponential_component=self.constants["damage_accumulation_exponential_component"],
                                                        cell_growth_rate=(1 - self.repair/self.constants["max_repair"])*self.constants["cell_growth_rate"]) * delta_t
        those_that_accumulate = n_array * np.abs(increment) / damage_step
        if (those_that_accumulate > n_array).sum() > 0:
            return None, False
        indices = (np.arange(len(n_array)) + np.divide(increment, np.abs(increment),
                                                       out=np.zeros_like(increment),
                                                        where=increment != 0)).round().astype(int)
        np.add.at(new_n_array,
                  indices[(0 <= indices) & (indices < len(n_array))],
                  those_that_accumulate[(0 <= indices) & (indices < len(n_array))])
        new_n_array += n_array - those_that_accumulate
        return new_n_array, True

    def reproduce(self, n_array: np.array, phi: float, delta_t: float, run: bool) -> (np.array, bool):
        if not run:
            return None, False
        division_rate = self.constants["cell_growth_rate"] * phi * (1 - self.repair / self.constants["max_repair"])
        if division_rate * delta_t > 1:
            return None, False

        new_n_array = np.zeros_like(n_array)
        for new_rho_vector in [self.n_array_bins * (1 + self.asymmetry), self.n_array_bins * (1 - self.asymmetry)]:
            indices = (new_rho_vector * (len(n_array) - 1)).round().astype(int)
            np.add.at(new_n_array,
                      indices[(0 <= indices) & (indices < len(n_array))],
                      n_array[(0 <= indices) & (indices < len(n_array))] * division_rate * delta_t)
        new_n_array += n_array * (1 - division_rate * delta_t)
        return new_n_array, True

    @property
    def mean_damage_concentration(self):
        return (self.n_array * self.n_array_bins).sum() / self.n_array.sum()

    @staticmethod
    def derivative_phi(n: float,
                       current_phi: float,
                       dilution_rate: float,
                       nutrient_acquisition_rate: float) -> float:
        nutrient_influx = dilution_rate * (1 - current_phi)
        nutrient_acquisition = nutrient_acquisition_rate * n * current_phi
        return nutrient_influx - nutrient_acquisition

    @staticmethod
    def derivative_rho(rho_vector: np.array,
                       current_phi: float,
                       damage_accumulation_linear_component: float,
                       damage_accumulation_exponential_component: float,
                       cell_growth_rate: float) -> float:
        damage_accumulation = damage_accumulation_linear_component + \
                                damage_accumulation_exponential_component * rho_vector
        damage_dilution = cell_growth_rate * current_phi * rho_vector
        return damage_accumulation - damage_dilution



class History:
    tables = {
        "history":
            {
                "columns":
                    {
                        "time_step": "INTEGER",
                        "n_cells": "INTEGER",
                        "mean_asymmetry": "REAL",
                        "mean_damage": "REAL",
                        "mean_repair": "REAL",
                        "environment": "BOOLEAN"
                    },
                "additional": ["PRIMARY KEY(time_step)"]
            },
        "cells":
            {
                "columns":
                    {
                        "record_id": "INTEGER primary key autoincrement",
                        "time_step": "INTEGER",
                        "cell_id": "INTEGER",
                        "cell_age": "INTEGER",
                        "cell_damage": "REAL",
                        "cell_asymmetry": "REAL",
                        "cell_damage_repair_intensity": "REAL",
                        "has_divided": "BOOLEAN",
                        "has_died": "TEXT"
                    },
                "additional": ["FOREIGN KEY(time_step) REFERENCES history (time_step)"]
            },
        "genealogy":
            {
                "columns":
                    {
                        "cell_id": "INTEGER",
                        "parent_id": "INTEGER",
                        "starting_damage": "REAL"
                    },
                "additional": ["PRIMARY KEY(cell_id)",
                               "FOREIGN KEY(cell_id) REFERENCES cells (cell_id)",
                               "FOREIGN KEY(parent_id) REFERENCES cells (cell_id)"]
            }
    }

    def __init__(self,
                 simulation_obj: SimulationManager,
                 save_path: str,
                 run_id: str,
                 run_name: str,
                 thread_id: int,
                 write_cells_table: bool, **kwargs):
        self.simulation_thread = simulation_obj
        self.run_id = run_id
        self.save_path = f"{save_path}/{self.run_id}{run_name}"
        self.thread_id = thread_id
        self.write_cells_table = False
        self.history_table, self.cells_table, self.genealogy_table = None, None, None
        self.traits_history = {}
        self.SQLdb = sqlite3.connect(f"{self.save_path}/{self.run_id}_{self.thread_id}.sqlite")
        # If the program exist with error, the connection will still be closed
        atexit.register(self.SQLdb.close)
        self.create_tables()
        self.reset()
        # This is needed not to record the same cell twice in the genealogy table
        self.max_cell_in_genealogy = -1

    def create_tables(self) -> None:
        for table in self.tables:
            if table == "cells" and not self.write_cells_table:
                continue
            columns = [" ".join([key, val]) for key, val in self.tables[table]["columns"].items()]
            content = ', '.join(columns)
            for element in self.tables[table]["additional"]:
                if "REFERENCES cells" in element and not self.write_cells_table:
                    continue
                content += ", " + element
            query = f"CREATE TABLE {table} ({content});"
            self.SQLdb.execute(query)
        self.SQLdb.commit()

    def reset(self) -> None:
        self.history_table, self.cells_table, self.genealogy_table = [], [], []

    def record(self, time_step) -> None:
        # Make a record history_table
        row = pd.DataFrame.from_dict(
            {"time_step": [time_step],
             "n_cells": [self.simulation_thread.current_population_size],
             "mean_asymmetry": self.simulation_thread.continuous_simulation.asymmetry,
             "mean_damage": self.simulation_thread.current_simulation.mean_damage_concentration,
             "mean_repair": self.simulation_thread.continuous_simulation.repair,
             "environment": self.simulation_thread.current_environment
             }
        )

        self.history_table.append(row)

        # Make a record in cells_table
        # if self.write_cells_table:
        #     cells = self.simulation_thread.chemostat.cells
        #     df_to_add = pd.DataFrame.from_dict(
        #         {"time_step": [time_step for _ in range(len(cells))],
        #          "cell_id": [cell.id for cell in cells],
        #          "cell_age": [cell.age for cell in cells],
        #          "cell_damage": [cell.damage for cell in cells],
        #          "cell_asymmetry": [cell.asymmetry for cell in cells],
        #          "cell_damage_repair_intensity": [cell.damage_repair_intensity for cell in cells],
        #          "has_divided": [bool(cell.has_reproduced) for cell in cells],
        #          "has_died": [cell.has_died for cell in cells]
        #          })
        #     df_to_add = df_to_add.reset_index(drop=True)
        #     self.cells_table.append(df_to_add)

        # Make a record in genealogy_table
        # cells = list(filter(lambda el: el.id > self.max_cell_in_genealogy,
        #                     self.simulation_thread.chemostat.cells))
        # if cells:
        #     df_to_add = pd.DataFrame.from_dict(
        #         {"cell_id": [cell.id for cell in cells],
        #          "parent_id": [cell.parent_id for cell in cells],
        #          "starting_damage": [cell.starting_damage for cell in cells]
        #          })
        #     self.genealogy_table.append(df_to_add)
        #     self.max_cell_in_genealogy = max(df_to_add.cell_id)

        # Make a record in traits
        # asymmetries = Counter([cell.asymmetry for cell in self.simulation_thread.chemostat.cells])
        # repairs = Counter([cell.damage_repair_intensity for cell in self.simulation_thread.chemostat.cells])
        # self.traits_history[time_step] = {
        #     "asymmetry": asymmetries,
        #     "repair": repairs
        # }

        # if len(self.cells_table) > 900 or len(self.history_table) > 2000:
        #     self.save()

        if len(self.history_table) > 2000:
            self.save()

    def save(self) -> None:
        for table, stem in zip([self.history_table, self.cells_table, self.genealogy_table],
                               ["history", "cells", "genealogy"]):
            if stem == "cells" and not self.write_cells_table:
                continue
            if table:
                table = pd.concat(table, ignore_index=True)
                table.to_sql(stem, self.SQLdb, if_exists='append', index=False)
        self.SQLdb.commit()
        self.reset()
        with open(f"{self.save_path}/traits.json", "w") as fl:
            json.dump(self.traits_history, fl)

    def __str__(self):
        return str(self.history_table)


if __name__ == "__main__":
    description = """
    This simulator can simulate populations of cells in a chemostat (which has two parameters: volume and dilution rate;
    cells are diluted with a poisson process with rate = population_size * dilution rate/volume. 
    The simulation is mainly aimed at simulating the processes associated with somatic damage accumulation. So, the 
    cells accumulate damage at some rate that can be tweaked, when they divide, the daughter cells inherit some of this 
    damage (how equally damage is distributed among cells can also be tweaked in the asymmetry parameter; this trait 
    can also evolve if you introduce non-zero mutation rate and mutation step). Damage hinders cells ability to 
    reproduce and, actually, to live. The way it does so can also be varied.
    """
    parser = argparse.ArgumentParser(prog="Chemostat simulator",
                                     description=description)
    # Chemostat
    parser.add_argument("-v", "--volume", default=1000, type=float)
    parser.add_argument("-d", "--dilution_rate", default=1, type=float)

    # Basic cell growth
    parser.add_argument("-nca", "--nutrient_critical_amount", default=10.0, type=float)
    parser.add_argument("-nar", "--nutrient_accumulation_rate", default=1.0, type=float)
    parser.add_argument("-mr", "--medium_richness", default=1.0, type=float)
    parser.add_argument("-nvsf", "--nutrient_to_volume_scaling_factor", default=1.0, type=float)

    # Damage
    parser.add_argument("-dalc", "--damage_accumulation_linear_component", default=0, type=float,
                        help="Next_step_damage = current_step_damage * daec + dalc; "
                             " 0 <= dalc <= 1 (1 is the lethal threshold)")
    parser.add_argument("-daec", "--damage_accumulation_exponential_component", default=0, type=float,
                        help="Next_step_damage = current_step_damage * daec + dalc")
    parser.add_argument("-dsd", "--damage_survival_dependency", default=1, type=float,
                        help="For each cell at each time step "
                             "P(death) = current_damage^damage_survival_dependency; "
                             "0 <= dsd <= inf. "
                             "To get threshold-like dependency set dsd close to 0. To punish even for minimal damage "
                             "set dsd to higher values.")
    parser.add_argument("-drd", "--damage_reproduction_dependency",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="if True, the expected age of reproduction is inversely proportional to the amount of "
                             "accumulated damage, if False no dependency")

    # Asymmetry
    parser.add_argument("-a", "--asymmetry", default=0, type=float)
    parser.add_argument("-amr", "--asymmetry_mutation_rate", default=0, type=float)
    parser.add_argument("-ams", "--asymmetry_mutation_step", default=0, type=float)

    # Repair
    parser.add_argument("-dri", "--damage_repair_intensity", default=0, type=float)
    parser.add_argument("-rm", "--repair_mode", default="additive", type=str,
                        choices=["additive", "multiplicative"])
    parser.add_argument("-rmr", "--repair_mutation_rate", default=0, type=float)
    parser.add_argument("-rms", "--repair_mutation_step", default=0, type=float)
    parser.add_argument("-rcc", "--max_repair", default=1, type=float)

    # Changing environment
    parser.add_argument("-chep", "--changing_environment_probability", default=0.0, type=float)
    parser.add_argument("-hef", "--harsh_environment_frac", default=1.0, type=float)

    # Technical
    parser.add_argument("-ni", "--niterations", default=100000, type=int)
    parser.add_argument("-nt", "--nthreads", default=1, type=int)
    parser.add_argument("-np", "--nprocs", default=1, type=int)
    parser.add_argument("-m", "--mode", default="local", type=str, choices=["cluster", "local", "interactive"])
    # noinspection PyTypeChecker
    parser.add_argument("--cells_table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--from_json", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--add_runs", type=str, default="")
    parser.add_argument("--deterministic_threshold", type=int, default=3000)
    parser.add_argument("--smart_initialization", action='store_true')

    args = parser.parse_args()

    if args.mode in ["local", "interactive"]:
        from tqdm import tqdm
        import multiprocessing

        save_path = "../data/local_experiments"
        if args.mode == "interactive":
            from interactive_mode import Drawer
    else:
        Path("../data/").mkdir(exist_ok=True)
        save_path = "../data"

    if args.add_runs:
        args.add_runs = f"{save_path}/{args.add_runs}"
        if not Path(args.add_runs).exists():
            raise FileNotFoundError(f"{args.add_runs} folder does not exist. "
                                    f"Please, supply --add_runs option with an existing folder.")
        args.from_json = f"{args.add_runs.rstrip('/')}/params.json"
        args.run_name = "_".join(Path(args.add_runs).stem.split("_")[1:])

    if args.from_json:
        logging.info(f"Reading arguments from {args.from_json}.")
        logging.info(f"All the arguments from the command line meaningful for the simulation contents will be ignored.")
        with open(args.from_json, "r") as fl:
            parameters = json.load(fl)
    else:
        parameters = {
            "chemostat_parameters": {
                "volume": args.volume,
                "dilution_rate": args.dilution_rate
            },
            "cell_parameters": {
                "damage_repair_mode": args.repair_mode,
                "damage_accumulation_linear_component": args.damage_accumulation_linear_component,
                "damage_accumulation_exponential_component": args.damage_accumulation_exponential_component,
                "damage_survival_dependency": args.damage_survival_dependency,
                "damage_reproduction_dependency": args.damage_reproduction_dependency,
                "nutrient_accumulation_rate": args.nutrient_accumulation_rate,
                "critical_nutrient_amount": args.nutrient_critical_amount,
                "nutrient_to_volume_scaling_factor": args.nutrient_to_volume_scaling_factor,
                "asymmetry_mutation_rate": args.asymmetry_mutation_rate,
                "asymmetry_mutation_step": args.asymmetry_mutation_step,
                "repair_mutation_rate": args.repair_mutation_rate,
                "repair_mutation_step": args.repair_mutation_step,
                "max_repair": args.max_repair
            },
            "asymmetry": args.asymmetry,
            "damage_repair_intensity": args.damage_repair_intensity,
            "changing_environment_probability": args.changing_environment_probability,
            "harsh_environment_frac": args.harsh_environment_frac,
            "medium_richness": args.medium_richness,

        }

    simulation = Simulation(parameters=parameters,
                            n_starting_cells=1,
                            save_path=save_path,
                            n_threads=args.nthreads,
                            n_procs=args.nprocs,
                            mode=args.mode,
                            write_cells_table=args.cells_table,
                            run_name=args.run_name,
                            add_runs=args.add_runs,
                            record_history=False,
                            deterministic_threshold=args.deterministic_threshold,
                            smart_initialization=args.smart_initialization
                            )
    simulation.run(args.niterations)
