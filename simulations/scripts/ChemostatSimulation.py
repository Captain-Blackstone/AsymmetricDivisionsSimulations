import random
import string
from wonderwords import RandomWord

import pandas as pd
import datetime
from pathlib import Path
from scipy.stats import gamma
import numpy as np
import os
import json
import argparse
import sqlite3
import atexit
import logging

from itertools import filterfalse
from collections import Counter

TIME_STEP_DURATION = 1
PRECISION = 3
logging.basicConfig(level=logging.INFO)


class Chemostat:
    def __init__(self, volume_val: float, dilution_rate: float, n_cells=None,
                 asymmetry=0.0, damage_repair_intensity=0.0):
        self.V = volume_val
        self.D = dilution_rate
        self._cells = []
        self._n = 0
        if n_cells:
            self.populate_with_cells(n_cells, asymmetry=asymmetry, damage_repair_intensity=damage_repair_intensity)

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

    def dilute(self) -> None:
        expected_n_cells_to_remove = self.D * self.N / self.V
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove*TIME_STEP_DURATION, 1)[0]
        dead_cells = np.random.choice(self.cells, size=min(self.N, n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die(cause="dilution")
        self._n = len(self._cells)

    @cells.setter
    def cells(self, cells):
        self._cells = cells
        self._n = len(self._cells)


class Cell:
    # Nutrient accumulation
    critical_nutrient_amount = 10
    nutrient_accumulation_rate = 1

    # Damage accumulation
    damage_accumulation_exponential_component = 0
    damage_accumulation_linear_component = 0.1
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
    damage_repair_mode = "additive"
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

    @staticmethod
    def lambda_large(x: float, alpha: float, beta=1.0) -> float:
        mu = alpha / beta  # expected value of the gamma-distribution
        age = mu * x  # cell age = expected age * (fraction of current age from the expected age)
        return mu * gamma.pdf(age, a=alpha, scale=1 / beta) / (1 - gamma.cdf(age, a=alpha, scale=1 / beta))

    @staticmethod
    def load_lookup_table_for_lambda_large() -> pd.DataFrame:
        """
        Generate a small lookup table for the probability of division that can be further extended.
        If a lookup table for the simulation parameters is already present in ../tables,
        load the table from file.
        :return:
        """

        def generate_the_table(lower_alpha: int, higher_alpha: int,
                               lower_x: float, higher_x: float, precision=PRECISION) -> pd.DataFrame:
            logging.info(f"Generating the probability of division lookup table for various critical nutrient amounts. "
                         f"Please, wait. We have {higher_alpha-lower_alpha} lines to process...")
            alphas = np.arange(lower_alpha, higher_alpha+1)
            xx = np.linspace(lower_x, higher_x, (higher_x-lower_x) * (10**precision) + 1)
            table = np.zeros((len(alphas), len(xx)))
            for i, alpha in enumerate(alphas):
                print("|", end="")
                for j, x in enumerate(xx):
                    table[i, j] = Cell.lambda_large(x, alpha)
            print(" Done")
            return pd.DataFrame(table, columns=[str(x) for x in xx.round(precision)], index=alphas)

        file = Path(Cell.lambda_large_lookup_path)
        if file.exists():
            lookup_table = pd.read_csv(str(file), index_col=0)
        else:
            lookup_table = generate_the_table(lower_alpha=1, higher_alpha=50, lower_x=0, higher_x=10)
        if Cell.critical_nutrient_amount not in lookup_table.index:
            added_lookup_table = generate_the_table(lower_alpha=max(lookup_table.index),
                                                    higher_alpha=Cell.critical_nutrient_amount,
                                                    lower_x=0, higher_x=10)
            lookup_table = pd.concat([lookup_table, added_lookup_table], ignore_index=True)
        return lookup_table

    @staticmethod
    def add_columns_to_lambda_large_lookup_table(x: float) -> None:
        max_current_x = float(Cell.lambda_large_lookup.columns[-1])
        while max_current_x < x:
            max_current_x += 1 / (10**PRECISION)
            max_current_x = round(max_current_x, PRECISION)
            Cell.lambda_large_lookup[str(max_current_x)] = \
                [Cell.lambda_large(max_current_x, alpha) for alpha in Cell.lambda_large_lookup.index]

    @staticmethod
    def archive_lookup_table() -> None:
        """
        Save the lookup table for the probability of cell division from its age and population size.
        Needed for subsequent runs - it's cheaper to load the table than to generate it from scratch.
        :return:
        """
        Cell.lambda_large_lookup.to_csv(Cell.lambda_large_lookup_path, sep=",")

    def prob_of_division_from_lookup_table(self) -> float:
        rate_of_uptake = ((1 - self.damage) ** Cell.damage_reproduction_dependency) * \
                         Cell.nutrient_accumulation_rate / self.chemostat.N
        if self._damage_repair_intensity > 0:
            rate_of_uptake *= (1 - self._damage_repair_intensity/Cell.damage_accumulation_linear_component) **\
                              Cell.repair_cost_coefficient
        if rate_of_uptake <= 0:
            return 0
        mu = Cell.critical_nutrient_amount / rate_of_uptake
        x = round(self.age / mu, PRECISION)
        if str(x) not in Cell.lambda_large_lookup.columns:
            Cell.add_columns_to_lambda_large_lookup_table(x)
        lambda_small = Cell.lambda_large_lookup.loc[Cell.critical_nutrient_amount, str(x)] / mu
        return TIME_STEP_DURATION * lambda_small

    def live(self) -> None:
        self._age += 1
        self.damage += (Cell.damage_accumulation_exponential_component * self._damage
                        + Cell.damage_accumulation_linear_component) \
            * TIME_STEP_DURATION
        if Cell.damage_repair_mode == "multiplicative":
            self.damage *= 1 - self._damage_repair_intensity
        elif Cell.damage_repair_mode == "additive":
            self.damage -= self.damage_repair_intensity * TIME_STEP_DURATION
        if np.random.uniform(0, 1) < self.damage**Cell.damage_survival_dependency:
            self.die(cause="damage")

    def reproduce(self, offspring_id: int) -> list:
        self._has_reproduced = np.random.uniform(0, 1) < self.prob_of_division_from_lookup_table()
        if self.has_reproduced:
            offspring_asymmetry = self.asymmetry
            if np.random.uniform() < self.asymmetry_mutation_rate:
                offspring_asymmetry += np.random.choice([self.asymmetry_mutation_step, -self.asymmetry_mutation_step])
            offspring_damage_repair_intensity = self.damage_repair_intensity
            if np.random.uniform() < self.repair_mutation_rate:
                offspring_damage_repair_intensity += np.random.choice([self.repair_mutation_step,
                                                                       -self.repair_mutation_step])
            res = [Cell(chemostat=self.chemostat,
                        cell_id=offspring_id,
                        parent_id=self.id,
                        asymmetry=offspring_asymmetry,
                        damage=self.damage * (1 + self.asymmetry) / 2,
                        damage_repair_intensity=offspring_damage_repair_intensity
                        ),
                   Cell(chemostat=self.chemostat,
                        cell_id=offspring_id + 1,
                        parent_id=self.id,
                        asymmetry=offspring_asymmetry,
                        damage=self.damage * (1 - self.asymmetry) / 2,
                        damage_repair_intensity=offspring_damage_repair_intensity
                        ),
                   ]
            return res
        else:
            res = [self]
            return res

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


class Simulation:
    def __init__(self,
                 parameters: dict,
                 n_starting_cells: int,
                 save_path: str,
                 mode: str,
                 n_threads: int, n_procs: int,
                 run_name="",
                 write_cells_table=False,
                 add_runs="", record_history=True):
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
        Cell.repair_cost_coefficient = parameters["cell_parameters"]["repair_cost_coefficient"]

        Cell.lambda_large_lookup = Cell.load_lookup_table_for_lambda_large()
        if mode != "cluster":
            atexit.register(Cell.archive_lookup_table)
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
        self.threads = [SimulationThread(run_id=run_id, run_name=run_name,
                                         thread_id=i + 1,
                                         chemostat_obj=Chemostat(
                                             volume_val=parameters["chemostat_parameters"]["volume"],
                                             dilution_rate=parameters["chemostat_parameters"]["dilution_rate"],
                                             n_cells=n_starting_cells,
                                             asymmetry=parameters["asymmetry"],
                                             damage_repair_intensity=parameters["damage_repair_intensity"]
                                         ),
                                         changing_environent_prob=parameters["changing_environment_probability"],
                                         save_path=save_path,
                                         mode=mode,
                                         write_cells_table=write_cells_table,
                                         record_history=record_history) for i in
                        range(max_existing_thread, max_existing_thread + n_threads)]
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


class SimulationThread:
    def __init__(self,
                 run_id: str,
                 run_name: str,
                 thread_id: int,
                 chemostat_obj: Chemostat,
                 changing_environent_prob: float,
                 save_path: str,
                 mode: str,
                 write_cells_table: bool,
                 record_history=True):
        self.mode = mode
        self.chemostat = chemostat_obj
        self.changing_environment_prob = changing_environent_prob
        self.changing_environment_val = {
            True: Cell.damage_accumulation_exponential_component,
            False: 0
        }
        self.current_environment = True
        if self.mode == "interactive":
            self.drawer = Drawer(self)

        self.record_history = record_history
        if self.record_history:
            self.history = History(self,
                                   save_path=save_path,
                                   run_id=run_id,
                                   run_name=run_name,
                                   thread_id=thread_id,
                                   write_cells_table=write_cells_table)

    def step(self, step_number: int) -> None:
        # If there are no cells, don't do anything

        # Cells are diluted
        self.chemostat.dilute()

        # Alive cells reproduce
        new_cells = []
        for cell in filterfalse(lambda cell_obj: cell_obj.has_died, self.chemostat.cells):
            offspring_id = max([cell.id for cell in self.chemostat.cells] + [cell.id for cell in new_cells]) + 1
            new_cells += cell.reproduce(offspring_id)

        # History is recorded
        if self.record_history:
            self.history.record(step_number)
        # Move to the next time step
        self.chemostat.cells = new_cells

        # Time passes
        for cell in self.chemostat.cells:
            cell.live()

        # Switch environment
        self._change_environment()

    def run(self, n_steps: int) -> None:
        np.random.seed((os.getpid() * int(datetime.datetime.now().timestamp()) % 123456789))
        if self.mode in ["local", "interactive"]:
            iterator = tqdm(range(n_steps))
        else:
            iterator = range(n_steps)

        for step_number in iterator:
            self.step(step_number)
            if self.mode == "interactive":
                self.drawer.draw_step(step_number)
            if self.chemostat.N == 0:
                logging.info("The population died out.")
                break
        if self.record_history:
            self.history.save()
            self.history.SQLdb.close()

    def _change_environment(self) -> None:
        if self.changing_environment_prob and random.uniform(0, 1) < self.changing_environment_prob:  # speed up
            self.current_environment = not self.current_environment
            Cell.damage_accumulation_exponential_component = self.changing_environment_val[self.current_environment]


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
                 simulation_obj: SimulationThread,
                 save_path: str,
                 run_id: str,
                 run_name: str,
                 thread_id: int,
                 write_cells_table: bool):
        self.simulation_thread = simulation_obj
        self.run_id = run_id
        self.save_path = f"{save_path}/{self.run_id}{run_name}"
        self.thread_id = thread_id
        self.write_cells_table = write_cells_table
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
             "n_cells": [self.simulation_thread.chemostat.N],
             "mean_asymmetry": [np.array([cell.asymmetry for cell in self.simulation_thread.chemostat.cells]).mean()],
             "mean_damage": [np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).mean()],
             "mean_repair": [np.array([cell.damage_repair_intensity
                                       for cell in self.simulation_thread.chemostat.cells]).mean()],
             "environment": self.simulation_thread.current_environment
             }
        )

        self.history_table.append(row)

        # Make a record in cells_table
        if self.write_cells_table:
            cells = self.simulation_thread.chemostat.cells
            df_to_add = pd.DataFrame.from_dict(
                {"time_step": [time_step for _ in range(len(cells))],
                 "cell_id": [cell.id for cell in cells],
                 "cell_age": [cell.age for cell in cells],
                 "cell_damage": [cell.damage for cell in cells],
                 "cell_asymmetry": [cell.asymmetry for cell in cells],
                 "cell_damage_repair_intensity": [cell.damage_repair_intensity for cell in cells],
                 "has_divided": [bool(cell.has_reproduced) for cell in cells],
                 "has_died": [cell.has_died for cell in cells]
                 })
            df_to_add = df_to_add.reset_index(drop=True)
            self.cells_table.append(df_to_add)

        # Make a record in genealogy_table
        cells = list(filter(lambda el: el.id > self.max_cell_in_genealogy,
                            self.simulation_thread.chemostat.cells))
        if cells:
            df_to_add = pd.DataFrame.from_dict(
                {"cell_id": [cell.id for cell in cells],
                 "parent_id": [cell.parent_id for cell in cells],
                 "starting_damage": [cell.starting_damage for cell in cells]
                 })
            self.genealogy_table.append(df_to_add)
            self.max_cell_in_genealogy = max(df_to_add.cell_id)

        # Make a record in traits
        asymmetries = Counter([cell.asymmetry for cell in self.simulation_thread.chemostat.cells])
        repairs = Counter([cell.damage_repair_intensity for cell in self.simulation_thread.chemostat.cells])
        self.traits_history[time_step] = {
            "asymmetry": asymmetries,
            "repair": repairs
        }

        if len(self.cells_table) > 900 or len(self.history_table) > 2000:
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
    parser.add_argument("-nca", "--nutrient_critical_amount", default=10, type=float)
    parser.add_argument("-nar", "--nutrient_accumulation_rate", default=1, type=float)

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
                             "accumulation set dsd to higher values.")
    parser.add_argument("-drd", "--damage_reproduction_dependency", default=True, type=bool,
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
    parser.add_argument("-rcc", "--repair_cost_coefficient", default=1, type=float)

    # Changing environment
    parser.add_argument("-chep", "--changing_environment_probability", default=0, type=float)

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

    args = parser.parse_args()

    if args.mode in ["local", "interactive"]:
        from tqdm import tqdm
        import multiprocessing
        save_path = "../data/local_experiments"
        if args.mode == "interactive":
            from interactive_mode import Drawer
    else:
        Path("data/").mkdir(exist_ok=True)
        save_path = "./data"

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
                "asymmetry_mutation_rate": args.asymmetry_mutation_rate,
                "asymmetry_mutation_step": args.asymmetry_mutation_step,
                "repair_mutation_rate": args.repair_mutation_rate,
                "repair_mutation_step": args.repair_mutation_step,
                "repair_cost_coefficient": args.repair_cost_coefficient
            },
            "asymmetry": args.asymmetry,
            "damage_repair_intensity": args.damage_repair_intensity,
            "changing_environment_probability": args.changing_environment_probability
        }

    simulation = Simulation(parameters=parameters,
                            n_starting_cells=1,
                            save_path=save_path,
                            n_threads=args.nthreads,
                            n_procs=args.nprocs,
                            mode=args.mode,
                            write_cells_table=args.cells_table,
                            run_name=args.run_name,
                            add_runs=args.add_runs
                            )
    simulation.run(args.niterations)
