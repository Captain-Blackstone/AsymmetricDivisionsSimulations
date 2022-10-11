import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root_path = "/home/blackstone/PycharmProjects/NIH/AsymmetricDivisions/simulations/data/"


class Visualizator:
    def __init__(self,
                 root_path: str,
                 run_id: int):
        files = [file.stem for file in Path(root_path + str(run_id)).glob("*.tsv")]
        n_threads = max([int(file.split("_")[-1]) for file in files])
        self.history_tables = [pd.read_csv(root_path + f"{run_id}/history_{run_id}_{i}.tsv", sep="\t")
                               for i in range(1, n_threads+1)]
        self.cell_tables = [pd.read_csv(root_path + f"{run_id}/cells_{run_id}_{i}.tsv", sep="\t")
                            for i in range(1, n_threads+1)]
        self.genealogy_tables = [pd.read_csv(root_path + f"{run_id}/genealogy_{run_id}_{i}.tsv", sep="\t")
                                 for i in range(1, n_threads + 1)]

    def plot(self, x_feature, y_feature, show=True):
        yy_sorted = []
        yy_max = []
        yy_min = []
        yy_mean = []
        xx = self.history_tables[0][x_feature]
        for x in xx:
            yy = np.array([list(self.history_tables[i].loc[self.history_tables[i][x_feature] == x,
                                                           y_feature])[0] for i in range(len(self.history_tables))])

            yy_min.append(yy.min())
            yy_max.append(yy.max())
            yy_mean.append(yy.mean())
            yy_sorted.append(sorted(yy))
        for i in range(len(yy_sorted[0])):
            plt.fill_between(xx, [el[i] for el in yy_sorted], yy_mean, color="blue", alpha=0.2)
        # plt.fill_between(xx, yy_min, yy_max, color="blue", alpha=0.5)
        plt.plot(xx, yy_mean, color="red")

        # for history_table in self.history_tables:
        #     plt.plot(history_table[x_feature], history_table[y_feature], color="blue")
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        if show:
            plt.show()

    def plot_age_distribution(self):
        for cell_table in self.cell_tables:
            xx = list(range(cell_table.generation.min(), cell_table.generation.max()+1))
            for x in xx:
                yy = cell_table.loc[cell_table.generation == x].cell_age
                plt.hist(yy, alpha=x/xx[-1], color="blue")
        plt.show()

    def plot_growth_rate(self):
        for cell_table in self.cell_tables:
            xx = list(range(cell_table.time_step.min(), cell_table.time_step.max() + 1))
            yy = []
            for x in xx:
                denominator = cell_table.loc[(cell_table.time_step == x) & (~cell_table.has_died)]
                numerator = denominator.loc[denominator.has_divided]
                yy.append(len(numerator)/len(denominator))
            print(np.array(yy[30:]).mean())
            plt.plot(xx, yy)
        plt.show()

    def plot_mean_feature(self, feature, condition=True, show=True):
        for cell_table in self.cell_tables:
            yy = [cell_table.loc[(cell_table.time_step == time_step) & condition, feature].mean()
                  for time_step in cell_table.time_step.unique()]
            xx = [el for el in range(len(yy)) if yy[el] is not np.nan]
            yy = [y for y in yy if y is not np.nan]
            plt.plot(xx, yy, color="blue")
        plt.xlabel("time_step")
        plt.ylabel(f"mean {feature}")
        if show:
            plt.show()


folders = [int(str(p).split("/")[-1]) for p in Path(root_path).glob("*")]

visualizator = Visualizator(root_path=root_path, run_id=1665492807)
visualizator.plot("time_step", "n_cells", show=False)
# visualizator.plot_mean_feature("cell_damage", show=False)
# visualizator.plot_mean_feature("cell_age", show=False)

visualizator = Visualizator(root_path=root_path, run_id=max(folders))
# visualizator = Visualizator(root_path=root_path, run_id=1665232310)
visualizator.plot("time_step", "n_cells", show=False)
# visualizator.plot_mean_feature("cell_damage", show=False)
# visualizator.plot_mean_feature("cell_age", show=False)

# visualizator.plot_age_distribution()
# visualizator.plot_growth_rate()
# visualizator.plot_mean_feature("cell_damage")
# visualizator.plot_mean_feature("cell_age", visualizator.cell_table.has_divided == True, show=False)
# visualizator.plot_mean_feature("cell_age", show=False)
# visualizator.plot_mean_feature("cell_damage", visualizator.cell_table.has_divided == True, show=False)
plt.grid()
plt.show()

# visualizator.plot_mean_feature("cell_age", show=True)
