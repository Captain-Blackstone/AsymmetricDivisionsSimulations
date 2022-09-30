

"""

Next, please do write up a lyx file (if you don't mind) or otherwise just a plain text document specifying exactly what
you do in pseudocode, so we can both easily refer to the algorithm. Currently, the cell number is constant for many
time steps (I suppose because the dilution rate is 0.1% correct?) and then abruptly jumps, so I think the next order of
business is to take care of the synchronization.
"""

import pandas as pd
import sys
import datetime
from pathlib import Path
from scipy.stats import gamma
import numpy as np


class Chemostat:
    def __init__(self, volume_val: float, dilution_rate: float, n_cells=None):
        self.V = volume_val
        self.D = dilution_rate
        self.cells = []
        if n_cells:
            self.populate_with_cells(n_cells)

    @property
    def N(self) -> int:
        return len(self.cells)

    def populate_with_cells(self, n_cells: int) -> None:
        self.cells += [Cell(chemostat=self, cell_id=i) for i in range(n_cells)]

    def dilute(self) -> None:
        expected_n_cells_to_remove = self.D * self.N/self.V
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove, 1)[0]
        dead_cells = np.random.choice(self.cells, size=min(self.N, n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die()


class Cell:
    critical_amount = 10
    rate_of_accumulation = 1

    def __init__(self,
                 chemostat: Chemostat,
                 cell_id: int,
                 parent_id=None,
                 age=0):
        self.chemostat = chemostat
        self.id = cell_id
        self.parent_id = parent_id
        self.age = age
        self.has_reproduced = False
        self.has_died = False

    @property
    def prob_of_division(self) -> float:
        """
        Instantaneous probability of cell to accumulate a material accumulating at rate beta to an amount alpha-1
        at a given age
        """
        alpha = Cell.critical_amount + 1
        beta = Cell.rate_of_accumulation/self.chemostat.N
        return gamma.pdf(self.age, a=alpha, scale=1/beta) / max(1 - gamma.cdf(self.age, a=alpha, scale=1/beta),
                                                                gamma.pdf(self.age, a=alpha, scale=1/beta))

    def reproduce(self, offspring_id: int) -> list:
        self.has_reproduced = np.random.uniform(0, 1) < self.prob_of_division
        if self.has_reproduced:
            return [Cell(chemostat=self.chemostat, cell_id=offspring_id, parent_id=self.id),
                    Cell(chemostat=self.chemostat, cell_id=offspring_id+1, parent_id=self.id)]
        else:
            return [self]

    def die(self) -> None:
        self.has_died = True


class Simulation:
    def __init__(self, chemostat_obj: Chemostat, save_path: str):
        self.chemostat = chemostat_obj
        self.history = History(self, save_path)

    @property
    def current_max_id(self) -> int:
        return max([cell.id for cell in self.chemostat.cells])

    def step(self, step_number: int) -> None:
        # Cells are diluted
        self.chemostat.dilute()

        # Alive cells reproduce
        new_cells = []
        for cell in filter(lambda cell_obj: not cell_obj.has_died, self.chemostat.cells):
            new_cells += cell.reproduce(self.current_max_id+1)

        # History is recorded
        self.history.record(step_number)

        # Move to the next time step
        self.chemostat.cells = new_cells

        # Time passes
        for cell in self.chemostat.cells:
            cell.age += 1

    def run(self, n_steps: int) -> None:
        for step_number in range(n_steps):
            self.step(step_number)
        self.history.save()


class History:
    def __init__(self, simulation_obj: Simulation, save_path: str):
        self.simulation = simulation_obj

        self.save_path = save_path
        self.run_id = round(datetime.datetime.now().timestamp())
        (Path(self.save_path) / Path(str(self.run_id))).mkdir()

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
            {"time_step": [time_step], "n_cells": [self.simulation.chemostat.N]})
        self.history_table = pd.concat([self.history_table, row], ignore_index=True)

        # Make a record in cells_table
        cells = self.simulation.chemostat.cells
        df_to_add = pd.DataFrame.from_dict(
            {"time_step": [time_step for _ in range(len(cells))],
             "cell_id": [cell.id for cell in cells],
             "cell_age": [cell.age for cell in cells],
             "has_divided": [cell.has_reproduced for cell in cells],
             "has_died": [cell.has_died for cell in cells]
             })

        self.cells_table = pd.concat([self.cells_table, df_to_add], ignore_index=True)

        # Make a record in genealogy_table
        cells = list(filter(lambda el: el.id not in self.genealogy_table.cell_id, self.simulation.chemostat.cells))
        df_to_add = pd.DataFrame.from_dict(
            {"cell_id": [cell.id for cell in cells],
             "parent_id": [cell.parent_id for cell in cells],
             })
        self.genealogy_table = pd.concat([self.genealogy_table, df_to_add], ignore_index=True)

        if len(self.cells_table) > 10000:
            self.save()

    def save(self) -> None:
        for table, stem in zip([self.history_table, self.cells_table, self.genealogy_table],
                               ["history", "cells", "genealogy"]):
            path = Path(f"{self.save_path}/{self.run_id}/{stem}_{self.run_id}.tsv")
            if path.exists():
                table.to_csv(path, mode="a", sep="\t", index=False, header=False)
            else:
                table.to_csv(path, sep="\t", index=False)
        self.reset()

    def __str__(self):
        return str(self.history_table)


if __name__ == "__main__":
    volume, dilution_rate = map(int, sys.argv[1:3])
    simulation = Simulation(Chemostat(volume_val=volume,
                                      dilution_rate=dilution_rate,
                                      n_cells=1),
                            save_path="../data/")
    simulation.run(10000)
