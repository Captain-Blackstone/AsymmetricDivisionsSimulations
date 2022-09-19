import numpy as np
import pandas as pd
import sys


class History:
    def __init__(self, simulation_obj):
        self.simulation = simulation_obj
        self.table = pd.DataFrame(columns=["generation", "n_cells"])

    def record(self):
        row = pd.DataFrame.from_dict({"generation": [len(self.table)+1], "n_cells": [self.simulation.chemostat.N]})
        self.table = pd.concat([self.table, row], ignore_index=True)

    def save(self, file_path):
        self.table.to_csv(file_path, sep="\t", index=False)

    def __str__(self):
        return str(self.table)


class Simulation:
    def __init__(self, chemostat_obj, carrying_capacity):
        self.chemostat = chemostat_obj
        self.density_coef = self._calculate_density_coef(carrying_capacity,
                                                         self.chemostat.dilution_rate,
                                                         self.chemostat.V)
        self.history = History(self)

    @staticmethod
    def _calculate_density_coef(carrying_capacity, dilution_rate, volume):
        return carrying_capacity/(np.log(1/2)/(2*np.log(1-dilution_rate/volume)))

    def step(self):
        # Cells reproduce
        new_cells = []
        for cell in self.chemostat.cells:
            new_cells += cell.reproduce()
        self.chemostat.cells = new_cells

        # Cells are diluted
        for _ in range(int(round(self.chemostat.N/self.density_coef))):
            self.chemostat.dilute()

            # History is recorded
        self.history.record()
        
    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()


class Chemostat:
    def __init__(self, volume, dilution_rate, n_cells=None):
        self.V = volume
        self.dilution_rate = dilution_rate
        self.cells = []
        if n_cells:
            self.populate_with_cells(n_cells)

    @property
    def N(self):
        return len(self.cells)

    def populate_with_cells(self, n_cells):
        self.cells += [Cell() for _ in range(n_cells)]

    def dilute(self):
        expected_n_cells_to_remove = self.dilution_rate * self.N/self.V
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove, 1)[0]
        if n_cells_to_remove > self.N:
            self.cells = []
        else:
            self.cells = np.random.choice(self.cells, size=self.N-n_cells_to_remove, replace=False)


class Cell:
    def __init__(self):
        pass

    @staticmethod
    def reproduce():
        return [Cell(), Cell()]


if __name__ == "__main__":
    volume, dilution_rate, carrying_capacity = map(int, sys.argv[1:4])
    simulation = Simulation(Chemostat(volume=volume,
                                      dilution_rate=dilution_rate,
                                      n_cells=1),
                            carrying_capacity=carrying_capacity)
    simulation.run(200)
    simulation.history.save("history.tsv")
