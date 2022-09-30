import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
root_path = "/home/blackstone/PycharmProjects/NIH/AsymmetricDivisions/simulations/data/"


class Visualizator:
    def __init__(self, root_path, run_id):
        self.history_table = pd.read_csv(root_path + f"{run_id}/history_{run_id}.tsv", sep="\t")
        self.cell_table = pd.read_csv(root_path + f"{run_id}/cells_{run_id}.tsv", sep="\t")

    def plot(self, x_feature, y_feature):
        plt.plot(self.history_table[x_feature], self.history_table[y_feature])
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()

    def plot_age_distribution(self):
        xx = list(range(self.cell_table.generation.min(), self.cell_table.generation.max()+1))
        for x in xx:
            yy = self.cell_table.loc[self.cell_table.generation == x].cell_age
            plt.hist(yy, alpha=x/xx[-1], color="blue")
        plt.show()

    def plot_growth_rate(self):
        xx = list(range(self.cell_table.generation.min(), self.cell_table.generation.max() + 1))
        yy = []
        for x in xx:
            denominator = self.cell_table.loc[(self.cell_table.generation == x) & (~self.cell_table.has_died)]
            numerator = denominator.loc[denominator.has_divided]
            yy.append(len(numerator)/len(denominator))
        print(np.array(yy[30:]).mean())
        plt.plot(xx, yy)
        plt.show()

from pathlib import Path
folders = [int(str(p).split("/")[-1]) for p in Path(root_path).glob("*")]
visualizator = Visualizator(root_path=root_path, run_id=max(folders))
visualizator.plot("time_step", "n_cells")
# visualizator.plot_age_distribution()
# visualizator.plot_growth_rate()