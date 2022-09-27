

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
    def __init__(self, volume_val: float, dilution_rate_val: float, n_cells=None):
        self.V = volume_val
        self.dilution_rate = dilution_rate_val
        self.cells = []
        if n_cells:
            self.populate_with_cells(n_cells)

    @property
    def N(self) -> int:
        return len(self.cells)

    def populate_with_cells(self, n_cells: int) -> None:
        self.cells += [Cell(i) for i in range(n_cells)]

    def dilute(self) -> None:
        expected_n_cells_to_remove = self.dilution_rate * (self.N**2)/self.V
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove, 1)[0]
        dead_cells = np.random.choice(self.cells, size=min(self.N, n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die()


class Cell:
    critical_amount = 10
    rate_of_accumulation = 1

    def __init__(self, cell_id: int, age=0):
        self.id = cell_id
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
        beta = Cell.rate_of_accumulation
        return gamma.pdf(self.age, a=alpha, scale=1/beta) / max(1 - gamma.cdf(self.age, a=alpha, scale=1/beta),
                                                                gamma.pdf(self.age, a=alpha, scale=1/beta))

    def reproduce(self, offspring_id: int) -> list:
        self.has_reproduced = np.random.uniform(0, 1) < self.prob_of_division
        if self.has_reproduced:
            return [Cell(self.id), Cell(offspring_id)]
        else:
            return [self]

    def die(self) -> None:
        self.has_died = True


class Simulation:
    def __init__(self, chemostat_obj: Chemostat, carrying_capacity_val: int):
        self.chemostat = chemostat_obj
        self.density_coefficient = self._calculate_density_coefficient(carrying_capacity_val,
                                                                       self.chemostat.dilution_rate,
                                                                       self.chemostat.V)
        self.history = History(self)

    @property
    def current_max_id(self) -> int:
        return max([cell.id for cell in self.chemostat.cells])

    @staticmethod
    def _calculate_density_coefficient(carrying_capacity_val: int, dilution_rate_val: float, volume_val: float) -> float:
        return carrying_capacity_val/(np.log(1/2)/(2*np.log(1-dilution_rate_val/volume_val)))

    def step(self) -> None:
        # Cells are diluted
        # for _ in range(int(round(self.chemostat.N/self.density_coefficient))):
        self.chemostat.dilute()

        # Alive cells reproduce
        new_cells = []
        for cell in filter(lambda cell_obj: not cell_obj.has_died, self.chemostat.cells):
            new_cells += cell.reproduce(self.current_max_id+1)

        # History is recorded
        self.history.record()

        # Move to the next generation
        self.chemostat.cells = new_cells

        # Time passes
        for cell in self.chemostat.cells:
            cell.age += 1

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()


class History:
    def __init__(self, simulation_obj: Simulation):
        self.simulation = simulation_obj
        self.history_table = pd.DataFrame(columns=["generation", "n_cells"])
        self.cells_table = pd.DataFrame(columns=["generation", "cell_id", "cell_age",
                                                 "cell_damage", "has_divided", "has_died"])

    def record(self) -> None:
        # Make a record history_table
        row = pd.DataFrame.from_dict(
            {"generation": [len(self.history_table) + 1], "n_cells": [self.simulation.chemostat.N]})
        self.history_table = pd.concat([self.history_table, row], ignore_index=True)

        # Make a record in cells_table
        cells = self.simulation.chemostat.cells
        df_to_add = pd.DataFrame.from_dict(
            {"generation": [len(self.history_table) + 1 for _ in range(len(cells))],
             "cell_id": [cell.id for cell in cells],
             "cell_age": [cell.age for cell in cells],
             "has_divided": [cell.has_reproduced for cell in cells],
             "has_died": [cell.has_died for cell in cells]
             })

        self.cells_table = pd.concat([self.cells_table, df_to_add], ignore_index=True)

    def save(self, output_folder: str) -> None:
        run_id = round(datetime.datetime.now().timestamp() % 1000000000)
        subfolder = Path(output_folder)/Path(str(run_id))
        subfolder.mkdir()
        history_file_path = subfolder/Path(f"history_{run_id}.tsv")
        cells_file_path = subfolder/Path(f"cells_{run_id}.tsv")
        for table, path in zip([self.history_table, self.cells_table],
                               [history_file_path, cells_file_path]):
            table.to_csv(path, sep="\t", index=False)

    def __str__(self):
        return str(self.history_table)


if __name__ == "__main__":
    volume, dilution_rate, carrying_capacity = map(int, sys.argv[1:4])
    simulation = Simulation(Chemostat(volume_val=volume,
                                      dilution_rate_val=dilution_rate,
                                      n_cells=1),
                            carrying_capacity_val=carrying_capacity)
    simulation.run(500)
    simulation.history.save("../data/")
