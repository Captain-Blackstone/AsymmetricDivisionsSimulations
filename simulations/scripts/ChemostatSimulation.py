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
    damage_accumulation_rate = 0
    damage_accumulation_intercept = 0.1
    lethal_damage_threshold = 1000

    # mutation rate
    mutation_rate = 0
    mutation_step = 0.01

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
        self._damage += Cell.damage_accumulation_rate*self._damage + Cell.damage_accumulation_intercept
        if self.damage > Cell.lethal_damage_threshold:
            self.die(cause="damage")

    def reproduce(self, offspring_id: int) -> list:
        self._has_reproduced = np.random.uniform(0, 1) < self.prob_of_division
        if self.has_reproduced:
            offspring_asymmetry = self.asymmetry
            if np.random.uniform() < self.mutation_rate:
                offspring_asymmetry += np.random.choice([self.mutation_step, -self.mutation_step])
            return [Cell(chemostat=self.chemostat,
                         cell_id=offspring_id,
                         parent_id=self.id,
                         asymmetry=offspring_asymmetry,
                         damage=self.damage * (1 + self.asymmetry) / 2),
                    Cell(chemostat=self.chemostat,
                         cell_id=offspring_id + 1,
                         parent_id=self.id,
                         asymmetry=offspring_asymmetry,
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
                 mode: str,
                 n_threads=1, n_procs=1):
        Cell.damage_accumulation_rate = parameters["cell_parameters"]["damage_accumulation_rate"]
        Cell.lethal_damage_threshold = parameters["cell_parameters"]["lethal_damage_threshold"]
        Cell.nutrient_accumulation_rate = parameters["cell_parameters"]["nutrient_accumulation_rate"]
        Cell.critical_nutrient_amount = parameters["cell_parameters"]["critical_nutrient_amount"]
        Cell.damage_accumulation_intercept = parameters["cell_parameters"]["damage_accumulation_intercept"]
        Cell.mutation_step = parameters["cell_parameters"]["mutation_step"]
        Cell.mutation_rate = parameters["cell_parameters"]["mutation_rate"]

        run_id = round(datetime.datetime.now().timestamp()*1000000)
        self.threads = [SimulationThread(run_id=run_id, thread_id=i + 1,
                                         chemostat_obj=Chemostat(
                                             volume_val=parameters["chemostat_parameters"]["volume"],
                                             dilution_rate=parameters["chemostat_parameters"]["dilution_rate"],
                                             n_cells=n_starting_cells,
                                             asymmetry=parameters["asymmetry"]),
                                         save_path=save_path,
                                         mode=mode) for i in range(n_threads)]
        self.n_procs = n_procs if mode == "local" else 1

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
    def __init__(self, run_id: int, thread_id: int, chemostat_obj: Chemostat, save_path: str, mode: str):
        self.mode = mode
        self.chemostat = chemostat_obj
        self.history = History(self,
                               save_path=save_path,
                               run_id=run_id,
                               thread_id=thread_id)

    def _step(self, step_number: int) -> None:
        # Cells are diluted
        self.chemostat.dilute()

        # Alive cells reproduce
        new_cells = []
        for cell in filter(lambda cell_obj: not cell_obj.has_died, self.chemostat.cells):
            offspring_id = max([cell.id for cell in self.chemostat.cells] + [cell.id for cell in new_cells]) + 1
            new_cells += cell.reproduce(offspring_id)

        # History is recorded
        self.history.record(step_number)

        # Move to the next time step
        self.chemostat.cells = new_cells

        # Time passes
        for cell in self.chemostat.cells:
            cell.live()

    def run(self, n_steps: int) -> None:
        np.random.seed((os.getpid() * int(datetime.datetime.now().timestamp()) % 123456789))
        if self.mode == "local":
            for step_number in tqdm(range(n_steps)):
                self._step(step_number)
                # if self.chemostat.N:
                #     print(self.chemostat.N, np.array([cell.damage for cell in self.chemostat.cells]).mean())
        elif self.mode == "cluster":
            for step_number in range(n_steps):
                self._step(step_number)
        self.history.save()
        self.history.SQLdb.close()


class History:
    tables = {
        "history":
            {
                "columns":
                    {
                        "time_step": "INTEGER",
                        "n_cells": "INTEGER"
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
                        "has_divided": "BOOLEAN",
                        "has_died": "BOOLEAN"
                    },
                "additional": ["FOREIGN KEY(time_step) REFERENCES history (time_step)"]
            },
        "genealogy":
            {
                "columns":
                    {
                        "cell_id": "INTEGER",
                        "parent_id": "INTEGER"
                    },
                "additional": ["PRIMARY KEY(cell_id)",
                               "FOREIGN KEY(cell_id) REFERENCES cells (cell_id)",
                               "FOREIGN KEY(parent_id) REFERENCES cells (cell_id)"]
            }
    }

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
        self.SQLdb = sqlite3.connect(f"{self.save_path}/{self.run_id}_{thread_id}.sqlite")
        # If the program exist with error, the conenction will still be closed
        atexit.register(self.SQLdb.close)
        self.create_tables()
        self.reset()
        # This is needed not to record the same cell twice in the genealogy table
        self.max_cell_in_genealogy = -1

    def create_tables(self) -> None:
        for table in self.tables:
            columns = [" ".join([key, val]) for key, val in self.tables[table]["columns"].items()]
            content = ', '.join(columns)
            for element in self.tables[table]["additional"]:
                content += ", " + element
            query = f"CREATE TABLE {table} ({content});"
            self.SQLdb.execute(query)
        self.SQLdb.commit()

    def reset(self) -> None:
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
        cells = list(filter(lambda el: el.id > self.max_cell_in_genealogy,
                            self.simulation_thread.chemostat.cells))
        if cells:
            df_to_add = pd.DataFrame.from_dict(
                {"cell_id": [cell.id for cell in cells],
                 "parent_id": [cell.parent_id for cell in cells],
                 })
            self.genealogy_table = pd.concat([self.genealogy_table, df_to_add], ignore_index=True)
            self.max_cell_in_genealogy = max(df_to_add.cell_id)
        if len(self.cells_table) > 5000:
            self.save()

    def save(self) -> None:
        for table, stem in zip([self.history_table, self.cells_table, self.genealogy_table],
                               ["history", "cells", "genealogy"]):

            table.to_sql(stem, self.SQLdb, if_exists='append', index=False)
            # path = Path(f"{self.save_path}/{stem}_{self.run_id}_{self.thread_id}.tsv")
            # if path.exists():
            #     table.to_csv(path, mode="a", sep="\t", index=False, header=False)
            # else:
            #     table.to_csv(path, sep="\t", index=False)
        self.SQLdb.commit()
        self.reset()

    def __str__(self):
        return str(self.history_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--asymmetry", default=0, type=float)
    parser.add_argument("-dai", "--damage_accumulation_intercept", default=0.1, type=float,
                        help="Next_step_damage = current_step_damage * dar + dai")
    parser.add_argument("-dar", "--damage_accumulation_rate", default=0, type=float,
                        help="Next_step_damage = current_step_damage * dar + dai")
    parser.add_argument("-dlr", "--damage_lethal_threshold", default=1000, type=float)
    parser.add_argument("-nca", "--nutrient_critical_amount", default=10, type=float)
    parser.add_argument("-nar", "--nutrient_accumulation_rate", default=1, type=float)
    parser.add_argument("-v", "--volume", default=1000, type=float)
    parser.add_argument("-d", "--dilution_rate", default=1, type=float)
    parser.add_argument("-mr", "--mutation_rate", default=0, type=float)
    parser.add_argument("-ms", "--mutation_step", default=0, type=float)
    parser.add_argument("-m", "--mode", default="cluster", type=str)
    args = parser.parse_args()
    parameters = {
        "chemostat_parameters": {
            "volume": args.volume,
            "dilution_rate": args.dilution_rate
        },
        "cell_parameters": {
            "damage_accumulation_intercept": args.damage_accumulation_intercept,
            "damage_accumulation_rate": args.damage_accumulation_rate,
            "lethal_damage_threshold": args.damage_lethal_threshold,
            "nutrient_accumulation_rate": args.nutrient_accumulation_rate,
            "critical_nutrient_amount": args.nutrient_critical_amount,
            "mutation_rate": args.mutation_rate,
            "mutation_step": args.mutation_step,
        },
        "asymmetry": args.asymmetry
    }
    if args.mode == "local":
        from tqdm import tqdm
        import multiprocessing

        save_path = "../data/local_experiments/"
    else:
        Path("data/").mkdir(exist_ok=True)
        save_path = "./data/"

    simulation = Simulation(parameters=parameters,
                            n_starting_cells=1,
                            save_path=save_path,
                            n_threads=1,
                            n_procs=1,
                            mode=args.mode)
    simulation.run(50000)
