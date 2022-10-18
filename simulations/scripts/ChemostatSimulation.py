import pandas as pd
import sys
import datetime
from pathlib import Path
from scipy.stats import gamma
import numpy as np
from tqdm import tqdm
import multiprocessing
import os
import json


class Chemostat:
    def __init__(self, volume_val: float, dilution_rate: float, n_cells=None, asymmetry=0):
        self.V = volume_val
        self.D = dilution_rate
        self._cells = []
        if n_cells:
            self.populate_with_cells(n_cells, asymmetry=asymmetry)

    @property
    def N(self) -> int:
        return len(self.cells)

    @property
    def cells(self) -> list:
        return self._cells

    def populate_with_cells(self, n_cells: int, asymmetry: float) -> None:
        self._cells += [Cell(chemostat=self, cell_id=i, asymmetry=asymmetry) for i in range(n_cells)]

    def dilute(self) -> None:
        expected_n_cells_to_remove = self.D * self.N / self.V
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove, 1)[0]
        dead_cells = np.random.choice(self.cells, size=min(self.N, n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die(cause="dilution")

    @cells.setter
    def cells(self, cells):
        self._cells = cells


class Cell:
    # Nutrient accumulation
    critical_nutrient_amount = 10
    nutrient_accumulation_rate = 1

    # Damage accumulation
    damage_accumulation_mode = "linear"
    damage_accumulation_rate = 0.1
    damage_accumulation_intercept = None
    lethal_damage_threshold = 1000

    def __init__(self,
                 chemostat: Chemostat,
                 cell_id: int,
                 parent_id=None,
                 asymmetry=0.0,
                 age=0,
                 damage=0.0):
        self.chemostat = chemostat
        self._id = cell_id
        self._parent_id = parent_id
        self._age = age
        self._damage = damage
        self._asymmetry = asymmetry
        self._has_reproduced = False
        self._has_died = ""

    @property
    def prob_of_division(self) -> float:
        """
        Instantaneous probability of cell to accumulate a material accumulating at rate beta to an amount alpha-1
        at a given age
        """
        alpha = Cell.critical_nutrient_amount + 1
        beta = Cell.nutrient_accumulation_rate / self.chemostat.N
        return gamma.pdf(self.age, a=alpha, scale=1 / beta) / max(1 - gamma.cdf(self.age, a=alpha, scale=1 / beta),
                                                                  gamma.pdf(self.age, a=alpha, scale=1 / beta))

    def live(self) -> None:
        self._age += 1
        if Cell.damage_accumulation_mode == "linear":
            self._damage += Cell.damage_accumulation_rate
        elif Cell.damage_accumulation_mode == "nonlinear":
            if Cell.damage_accumulation_intercept is not None:
                self._damage += Cell.damage_accumulation_rate*self._damage + Cell.damage_accumulation_intercept
            else:
                raise ValueError(
                    "Damage accumulation intercept cannot be None if damage accumulation mode is nonlinear"
                )
        else:
            raise ValueError("Unknown damage accumulation mode")
        if self.damage > Cell.lethal_damage_threshold:
            self.die(cause="damage")

    def reproduce(self, offspring_id: int) -> list:
        self._has_reproduced = np.random.uniform(0, 1) < self.prob_of_division
        if self.has_reproduced:
            return [Cell(chemostat=self.chemostat,
                         cell_id=offspring_id,
                         parent_id=self.id,
                         asymmetry=self.asymmetry,
                         damage=self.damage * (1 + self.asymmetry) / 2),
                    Cell(chemostat=self.chemostat,
                         cell_id=offspring_id + 1,
                         parent_id=self.id,
                         asymmetry=self.asymmetry,
                         damage=self.damage * (1 - self.asymmetry) / 2)]
        else:
            return [self]

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


class Simulation:
    def __init__(self,
                 parameters: dict,
                 n_starting_cells: int,
                 save_path: str,
                 n_threads=1, n_procs=1):
        Cell.damage_accumulation_rate = parameters["cell_parameters"]["damage_accumulation_rate"]
        Cell.lethal_damage_threshold = parameters["cell_parameters"]["lethal_damage_threshold"]
        Cell.nutrient_accumulation_rate = parameters["cell_parameters"]["nutrient_accumulation_rate"]
        Cell.critical_nutrient_amount = parameters["cell_parameters"]["critical_nutrient_amount"]
        Cell.damage_accumulation_mode = parameters["cell_parameters"]["damage_accumulation_mode"]
        Cell.damage_accumulation_intercept = parameters["cell_parameters"]["damage_accumulation_intercept"]

        run_id = round(datetime.datetime.now().timestamp())
        self.threads = [SimulationThread(run_id=run_id, thread_id=i + 1,
                                         chemostat_obj=Chemostat(
                                             volume_val=parameters["chemostat_parameters"]["volume"],
                                             dilution_rate=parameters["chemostat_parameters"]["dilution_rate"],
                                             n_cells=n_starting_cells,
                                             asymmetry=parameters["asymmetry"]),
                                         save_path=save_path) for i in range(n_threads)]
        self.n_procs = n_procs

        # Write parameters needed to identify simulation
        with open(f"{save_path}/{run_id}/params.txt", "w") as fl:
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
    def __init__(self, run_id: int, thread_id: int, chemostat_obj: Chemostat, save_path: str):
        self.chemostat = chemostat_obj
        self.history = History(self,
                               save_path=save_path,
                               run_id=run_id,
                               thread_id=thread_id)

    @property
    def current_max_id(self) -> int:
        return max([cell.id for cell in self.chemostat.cells])

    def _step(self, step_number: int) -> None:
        # Cells are diluted
        self.chemostat.dilute()

        # Alive cells reproduce
        new_cells = []
        for cell in filter(lambda cell_obj: not cell_obj.has_died, self.chemostat.cells):
            new_cells += cell.reproduce(self.current_max_id + 1)

        # History is recorded
        self.history.record(step_number)

        # Move to the next time step
        self.chemostat.cells = new_cells

        # Time passes
        for cell in self.chemostat.cells:
            cell.live()

    def run(self, n_steps: int) -> None:
        np.random.seed((os.getpid() * int(datetime.datetime.now().timestamp()) % 123456789))
        for step_number in tqdm(range(n_steps)):
            self._step(step_number)
        self.history.save()


class History:
    def __init__(self,
                 simulation_obj: SimulationThread,
                 save_path: str,
                 run_id: int,
                 thread_id: int):
        self.simulation_thread = simulation_obj
        self.run_id = run_id
        (Path(save_path) / Path(str(self.run_id))).mkdir(exist_ok=True)
        self.save_path = f"{save_path}/{self.run_id}"
        self.thread_id = thread_id
        self.history_table, self.cells_table, self.genealogy_table = None, None, None
        self.reset()

    def reset(self):
        self.history_table = pd.DataFrame(columns=["time_step", "n_cells"])
        self.cells_table = pd.DataFrame(columns=["time_step", "cell_id", "cell_age",
                                                 "cell_damage", "has_divided", "has_died"])
        self.genealogy_table = pd.DataFrame(columns=["cell_id", "parent_id"])

    def record(self, time_step) -> None:
        # Make a record history_table
        row = pd.DataFrame.from_dict(
            {"time_step": [time_step], "n_cells": [self.simulation_thread.chemostat.N]})
        self.history_table = pd.concat([self.history_table, row], ignore_index=True)

        # Make a record in cells_table
        cells = self.simulation_thread.chemostat.cells
        df_to_add = pd.DataFrame.from_dict(
            {"time_step": [time_step for _ in range(len(cells))],
             "cell_id": [cell.id for cell in cells],
             "cell_age": [cell.age for cell in cells],
             "cell_damage": [cell.damage for cell in cells],
             "has_divided": [cell.has_reproduced for cell in cells],
             "has_died": [cell.has_died for cell in cells]
             })

        self.cells_table = pd.concat([self.cells_table, df_to_add], ignore_index=True)

        # Make a record in genealogy_table
        cells = list(filter(lambda el: el.id not in self.genealogy_table.cell_id,
                            self.simulation_thread.chemostat.cells))
        df_to_add = pd.DataFrame.from_dict(
            {"cell_id": [cell.id for cell in cells],
             "parent_id": [cell.parent_id for cell in cells],
             })
        self.genealogy_table = pd.concat([self.genealogy_table, df_to_add], ignore_index=True)

        if len(self.cells_table) > 5000:
            self.save()

    def save(self) -> None:
        for table, stem in zip([self.history_table, self.cells_table, self.genealogy_table],
                               ["history", "cells", "genealogy"]):
            path = Path(f"{self.save_path}/{stem}_{self.run_id}_{self.thread_id}.tsv")
            if path.exists():
                table.to_csv(path, mode="a", sep="\t", index=False, header=False)
            else:
                table.to_csv(path, sep="\t", index=False)
        self.reset()

    def __str__(self):
        return str(self.history_table)


if __name__ == "__main__":
    asymmetry, damage_accumulation_mode, damage_accumulation_intercept, damage_accumulation_rate, \
        lethal_damage_threshold, critical_nutrient_amount, nutrient_accumulation_rate, \
        volume, dilution_rate = sys.argv[1:10]
    parameters = {
        "chemostat_parameters": {
            "volume": float(volume),
            "dilution_rate": float(dilution_rate)
        },
        "cell_parameters": {
            "damage_accumulation_mode": damage_accumulation_mode,
            "damage_accumulation_intercept": float(damage_accumulation_intercept),
            "damage_accumulation_rate": float(damage_accumulation_rate),
            "lethal_damage_threshold": float(lethal_damage_threshold),
            "nutrient_accumulation_rate": float(nutrient_accumulation_rate),
            "critical_nutrient_amount": float(critical_nutrient_amount),
        },
        "asymmetry": float(asymmetry)
    }
    simulation = Simulation(parameters=parameters, n_starting_cells=1, save_path="../data/", n_threads=8, n_procs=8)
    simulation.run(10000)
