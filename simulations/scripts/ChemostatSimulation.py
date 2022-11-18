import time

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
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, filename="log.log")


class Chemostat:
    def __init__(self, volume_val: float, dilution_rate: float, n_cells=None, asymmetry=0):
        self.V = volume_val
        self.D = dilution_rate
        self._cells = []
        self._n = 0
        if n_cells:
            self.populate_with_cells(n_cells, asymmetry=asymmetry)

    @property
    def N(self) -> int:
        return self._n

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

    def live(self) -> None:
        self._age += 1
        self._damage += Cell.damage_accumulation_exponential_component * self._damage + \
            Cell.damage_accumulation_linear_component
        if self.damage > Cell.lethal_damage_threshold:
            self.die(cause="damage")

    def reproduce(self, offspring_id: int) -> list:
        t1 = time.perf_counter()
        self._has_reproduced = np.random.uniform(0, 1) < get_prob_of_division(self.age, self.chemostat.N)
        t2 = time.perf_counter()
        if self.has_reproduced:
            offspring_asymmetry = self.asymmetry
            if np.random.uniform() < self.mutation_rate:
                offspring_asymmetry += np.random.choice([self.mutation_step, -self.mutation_step])
                offspring_asymmetry = min(max(offspring_asymmetry, 0), 1)
            t3 = time.perf_counter()
            res = [Cell(chemostat=self.chemostat,
                        cell_id=offspring_id,
                        parent_id=self.id,
                        asymmetry=offspring_asymmetry,
                        damage=self.damage * (1 + self.asymmetry) / 2),
                   Cell(chemostat=self.chemostat,
                        cell_id=offspring_id + 1,
                        parent_id=self.id,
                        asymmetry=offspring_asymmetry,
                        damage=self.damage * (1 - self.asymmetry) / 2)]
            t4 = time.perf_counter()
            return res
        else:
            res = [self]
            t3 = time.perf_counter()
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
        Cell.damage_accumulation_exponential_component = \
            parameters["cell_parameters"]["damage_accumulation_exponential_component"]
        Cell.damage_accumulation_linear_component = \
            parameters["cell_parameters"]["damage_accumulation_linear_component"]

        Cell.lethal_damage_threshold = parameters["cell_parameters"]["lethal_damage_threshold"]
        Cell.nutrient_accumulation_rate = parameters["cell_parameters"]["nutrient_accumulation_rate"]
        Cell.critical_nutrient_amount = parameters["cell_parameters"]["critical_nutrient_amount"]
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
        self.n_procs = n_procs if mode in ["local", "interactive"] else 1

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
    def __init__(self,
                 run_id: int,
                 thread_id: int,
                 chemostat_obj: Chemostat,
                 save_path: str,
                 mode: str):
        self.mode = mode
        self.chemostat = chemostat_obj
        self.history = History(self,
                               save_path=save_path,
                               run_id=run_id,
                               thread_id=thread_id)
        if self.mode == "interactive":
            self.drawer = Drawer(self)

    def _step(self, step_number: int) -> None:
        # Cells are diluted
        t1 = time.perf_counter()
        self.chemostat.dilute()
        t2 = time.perf_counter()
        logging.debug(f"{self.chemostat.N} {t2-t1: 0.5f}: dilution")
        # Alive cells reproduce
        new_cells = []
        for cell in filter(lambda cell_obj: not cell_obj.has_died, self.chemostat.cells):
            offspring_id = max([cell.id for cell in self.chemostat.cells] + [cell.id for cell in new_cells]) + 1
            new_cells += cell.reproduce(offspring_id)
        t3 = time.perf_counter()
        logging.debug(f"{self.chemostat.N} {t3-t2: 0.5f}: reproduction")
        # History is recorded
        self.history.record(step_number)
        # Move to the next time step
        self.chemostat.cells = new_cells
        t4 = time.perf_counter()
        # Time passes
        for cell in self.chemostat.cells:
            cell.live()
        t5 = time.perf_counter()
        logging.debug(f"{self.chemostat.N} {t4 - t3: 0.5f}: history recording")
        logging.debug(f"{self.chemostat.N} {t5 - t4: 0.5f}: live")
        logging.debug("----")
    def run(self, n_steps: int) -> None:
        np.random.seed((os.getpid() * int(datetime.datetime.now().timestamp()) % 123456789))
        if self.mode == "local":
            for step_number in tqdm(range(n_steps)):
                self._step(step_number)
        elif self.mode == "interactive":
            for step_number in tqdm(range(n_steps)):
                self._step(step_number)
                self.drawer.draw_step(step_number)
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
                        "cell_asymmetry": "REAL",
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
        # self.history_table = pd.DataFrame(columns=["time_step", "n_cells"])
        # self.cells_table = pd.DataFrame(columns=["time_step", "cell_id", "cell_age",
        #                                          "cell_damage", "cell_asymmetry", "has_divided", "has_died"])
        # self.cells_table["has_divided"] = self.cells_table["has_divided"].astype(bool)
        # self.genealogy_table = pd.DataFrame(columns=["cell_id", "parent_id"])
        self.history_table, self.cells_table, self.genealogy_table = [], [], []

    def record(self, time_step) -> None:
        # Make a record history_table
        t1 = time.perf_counter()
        row = pd.DataFrame.from_dict(
            {"time_step": [time_step], "n_cells": [self.simulation_thread.chemostat.N]})
        t2 = time.perf_counter()

        # self.history_table = pd.concat([self.history_table, row], ignore_index=True)
        self.history_table.append(row)
        t3 = time.perf_counter()

        # Make a record in cells_table
        cells = self.simulation_thread.chemostat.cells
        df_to_add = pd.DataFrame.from_dict(
            {"time_step": [time_step for _ in range(len(cells))],
             "cell_id": [cell.id for cell in cells],
             "cell_age": [cell.age for cell in cells],
             "cell_damage": [cell.damage for cell in cells],
             "has_divided": [bool(cell.has_reproduced) for cell in cells],
             "has_died": [cell.has_died for cell in cells]
             })
        df_to_add = df_to_add.reset_index(drop=True)
        t5 = time.perf_counter()


        # self.cells_table = pd.concat([self.cells_table, df_to_add], ignore_index=True)
        self.cells_table.append(df_to_add)
        t7 = time.perf_counter()

        # Make a record in genealogy_table
        cells = list(filter(lambda el: el.id > self.max_cell_in_genealogy,
                            self.simulation_thread.chemostat.cells))
        if cells:
            df_to_add = pd.DataFrame.from_dict(
                {"cell_id": [cell.id for cell in cells],
                 "parent_id": [cell.parent_id for cell in cells],
                 })
            # self.genealogy_table = pd.concat([self.genealogy_table, df_to_add], ignore_index=True)
            self.genealogy_table.append(df_to_add)
            self.max_cell_in_genealogy = max(df_to_add.cell_id)
        t12 = t7
        if len(self.cells_table) > 900:
            self.save()
            t12 = time.perf_counter()
        logging.debug(f"{self.simulation_thread.chemostat.N} {t2 - t1}: form history row")
        logging.debug(f"{self.simulation_thread.chemostat.N} {t3 - t2}: concatenate history row")
        logging.debug(f"{self.simulation_thread.chemostat.N} {t5 - t3}: form cells row ")
        logging.debug(f"{self.simulation_thread.chemostat.N} {t7 - t5}: concatenate cells row")
        logging.debug(f"{self.simulation_thread.chemostat.N} {t12 - t7}: save to sql")

    def save(self) -> None:
        for table, stem in zip([self.history_table, self.cells_table, self.genealogy_table],
                               ["history", "cells", "genealogy"]):

            table = pd.concat(table, ignore_index=True)
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


class Drawer:
    def __init__(self, simulation_thread: SimulationThread):
        self.simulation_thread = simulation_thread
        self.update_time = 100  # number of steps between figure updates
        self.resolution = 10  # number of steps between data collection events
        self.plot_how_many = 1000  # number of points present on the plot at each time point

        self.fig, self.ax = plt.subplots(3, 1)
        for i, title in enumerate(["Population size", "Mean damage", "Asymmetry"]):
            self.ax[i].set_title(title, fontsize=10)

        data_dicts = [
            {"ax_num": 0, "color": "blue", "alpha": 1,
             "update_function": lambda: self.simulation_thread.chemostat.N},
            {"ax_num": 1, "color": "green", "alpha": 1,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).mean()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 1, "color": "green", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).max()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 1, "color": "green", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).min()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 2,  "color": "red", "alpha": 1,
             "update_function":
                 lambda: np.array([cell.asymmetry for cell in self.simulation_thread.chemostat.cells]).mean()
                 if self.simulation_thread.chemostat.N else 0},
        ]

        self.plots = []
        for data_dict in data_dicts:
            self.plots.append(Plot(self,
                                   self.plot_how_many,
                                   self.ax[data_dict["ax_num"]],
                                   data_dict["color"],
                                   data_dict["alpha"],
                                   data_dict["update_function"]))

        plt.get_current_fig_manager().full_screen_toggle()

    def draw_step(self, step_number):
        # Collect data each self.resolution steps
        if step_number % self.resolution == 0:
            for plot in self.plots:
                plot.collect_data(step_number)
        # Update figure each self.update_time steps
        if step_number % self.update_time == 0:
            for plot in self.plots:
                plot.update_data()
            for plot in self.plots:
                plot.update_plot()
            self.fig.canvas.draw()
            plt.pause(0.01)


class Plot:
    def __init__(self,
                 drawer: Drawer,
                 plot_how_many: int,
                 ax: plt.Axes,
                 color: str,
                 alpha: str,
                 update_function):
        self.drawer, self.plot_how_many = drawer, plot_how_many
        self.ax, self.color, self.alpha = ax, color, alpha
        self.update_function = update_function
        self.xdata, self.ydata = [], []
        self.layer, = self.ax.plot(self.xdata, self.ydata, color=self.color, alpha=self.alpha)

    def collect_data(self, step_num: int):
        self.xdata.append(step_num)
        self.ydata.append(self.update_function())
        self.xdata = self.xdata[-self.plot_how_many:]
        self.ydata = self.ydata[-self.plot_how_many:]

    def update_data(self):
        self.layer.set_ydata(self.ydata)
        self.layer.set_xdata(self.xdata)

    def update_plot(self):
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)


# Lookup table for division probability
def prob_of_division(age, N) -> float:
    """
    Instantaneous probability of cell to accumulate a material accumulating at rate beta to an amount alpha-1
    at a given age
    """
    alpha = Cell.critical_nutrient_amount + 1
    beta = Cell.nutrient_accumulation_rate / N
    return gamma.pdf(age, a=alpha, scale=1 / beta) / max(1 - gamma.cdf(age, a=alpha, scale=1 / beta),
                                                         gamma.pdf(age, a=alpha, scale=1 / beta))


PROB_OF_DIVISION = np.array([prob_of_division(age, 1) for age in range(100)]).reshape(1, 100)


# Function for a cell to learn its division probability
def get_prob_of_division(age, N):
    global PROB_OF_DIVISION
    max_N, max_age = PROB_OF_DIVISION.shape
    for n in range(max_N+1, N+1):
        PROB_OF_DIVISION = np.vstack((PROB_OF_DIVISION,
                                      np.array([prob_of_division(a, n) for a in range(1, max_age+1)])))
        PROB_OF_DIVISION = np.nan_to_num(PROB_OF_DIVISION)
    max_N, max_age = PROB_OF_DIVISION.shape
    for a in range(max_age + 1, age + 1):
        PROB_OF_DIVISION = np.c_[PROB_OF_DIVISION, np.array([prob_of_division(a, n) for n in range(1, max_N+1)])]
        PROB_OF_DIVISION = np.nan_to_num(PROB_OF_DIVISION)
    return PROB_OF_DIVISION[N-1, age-1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--asymmetry", default=0, type=float)
    parser.add_argument("-dalc", "--damage_accumulation_linear_component", default=0.1, type=float,
                        help="Next_step_damage = current_step_damage * dar + dai")
    parser.add_argument("-daec", "--damage_accumulation_exponential_component", default=0, type=float,
                        help="Next_step_damage = current_step_damage * dar + dai")
    parser.add_argument("-dlt", "--damage_lethal_threshold", default=1000, type=float)
    parser.add_argument("-nca", "--nutrient_critical_amount", default=10, type=float)
    parser.add_argument("-nar", "--nutrient_accumulation_rate", default=1, type=float)
    parser.add_argument("-v", "--volume", default=1000, type=float)
    parser.add_argument("-d", "--dilution_rate", default=1, type=float)
    parser.add_argument("-mr", "--mutation_rate", default=0, type=float)
    parser.add_argument("-ms", "--mutation_step", default=0, type=float)
    parser.add_argument("-m", "--mode", default="cluster", type=str)
    parser.add_argument("-nt", "--nthreads", default=1, type=int)
    parser.add_argument("-np", "--nprocs", default=1, type=int)
    parser.add_argument("-ni", "--niterations", default=10000, type=int)

    args = parser.parse_args()
    parameters = {
        "chemostat_parameters": {
            "volume": args.volume,
            "dilution_rate": args.dilution_rate
        },
        "cell_parameters": {
            "damage_accumulation_linear_component": args.damage_accumulation_linear_component,
            "damage_accumulation_exponential_component": args.damage_accumulation_exponential_component,
            "lethal_damage_threshold": args.damage_lethal_threshold,
            "nutrient_accumulation_rate": args.nutrient_accumulation_rate,
            "critical_nutrient_amount": args.nutrient_critical_amount,
            "mutation_rate": args.mutation_rate,
            "mutation_step": args.mutation_step,
        },
        "asymmetry": args.asymmetry
    }
    if args.mode in ["local", "interactive"]:
        from tqdm import tqdm
        import multiprocessing

        save_path = "../data/local_experiments/"
    else:
        Path("data/").mkdir(exist_ok=True)
        save_path = "./data/"

    simulation = Simulation(parameters=parameters,
                            n_starting_cells=1,
                            save_path=save_path,
                            n_threads=args.nthreads,
                            n_procs=args.nprocs,
                            mode=args.mode)
    simulation.run(args.niterations)
