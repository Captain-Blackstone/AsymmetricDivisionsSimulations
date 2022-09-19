"""
Firstly, the simulation runs fine on the cluster. I would rename the output file so that the parameters are readable
within the file name - alternatively as a header for the output - whatever is your preference. I would also add a dummy
index variable so that the output always have unique names and can be saved in the same directory.

For the data itself, I would reformat so that each line contains: column 1 = timestep, column 2 = unique cell ID (the order in which the cell was simulated), column 3 = cell age, column 4 = cell damage, column 5 = true if cell divided in that time step, column 6 = true if cell died in that time step + whatever else we include later. The formatting is up to you of course but I think this is the minimum data we need to record.

Next, please do write up a lyx file (if you don't mind) or otherwise just a plain text document specifying exactly what you do in pseudo code so we can both easily refer to the algorithm. Currently the cell number is constant for many timesteps (I suppose because the dilution rate is 0.1% correct?) and then abruptly jumps so I think the next order of business is to take care of the synchronization.

I have attached some notes about how to address the synchronization. I'm familiar with the formalism without incorporating damage accumulation or asymmetric division. I started sketching out how to solve the case for damage accumulation but didn't get far. Do you still want to meet in the afternoon (let's say 1:15 this time if that's OK) or would you like to meet in the morning? If so, let's say 9:30am. I suspect you're asleep now and I will be asleep when you wake up, but just email me and I'll plan accordingly
"""
import numpy as np
import pandas as pd
import sys
import datetime


class History:
    def __init__(self, simulation_obj):
        self.simulation = simulation_obj
        self.table = pd.DataFrame(columns=["generation", "n_cells"])

    def record(self):
        row = pd.DataFrame.from_dict({"generation": [len(self.table)+1], "n_cells": [self.simulation.chemostat.N]})
        self.table = pd.concat([self.table, row], ignore_index=True)

    def save(self, file_path):
        file_path = ".".join(file_path.split(".")[:-1]) + f"_{round(datetime.datetime.now().timestamp() % 100000000)}." + file_path.split(".")[-1]
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
    simulation.history.save("../data/history.tsv")
