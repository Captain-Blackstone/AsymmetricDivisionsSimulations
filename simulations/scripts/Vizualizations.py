import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sqlite3
import atexit
import logging
root_path = "/home/blackstone/PycharmProjects/NIH/AsymmetricDivisions/simulations/data/local_experiments"
logging.basicConfig(level=logging.INFO)


class Visualizator:
    def __init__(self,
                 root_path: str,
                 run_id,
                 color="blue", label=""):
        if Path(root_path + f"/{run_id}/params.txt").exists():
            with open(root_path + f"/{run_id}/params.txt", "r") as fl:
                self.params = json.load(fl)
        else:
            with open(root_path + f"/{run_id}/params.json", "r") as fl:
                self.params = json.load(fl)
        self.color = color
        self.label = (label + f" ({run_id})").strip()
        self.history_tables = []
        self.cell_tables = []
        self.genealogy_tables = []

    def plot_single_run(self, history_table, y_feature, axis, label=None, convolution_step=100):
        yy = history_table[y_feature]
        yy = list(yy)[:int(convolution_step / 2)] + list(np.convolve(yy, np.ones(convolution_step) /
                                                                     convolution_step, 'valid'))
        if label is None:
            axis.plot(yy, color=self.color)
        else:
            axis.plot(yy, label=label, color=self.color)

    def plot(self, x_feature, y_feature, which=-1, show=True, axis=None, label=None):
        # yy_sorted = []
        # for x in xx:
        #     yy = np.array([list(self.history_tables[i].loc[self.history_tables[i][x_feature] == x,
        #                                                    y_feature])[0] for i in range(len(self.history_tables))])
        #     yy_sorted.append(np.array(sorted(yy)))
        # plt.fill_between(xx,
        #                  [el[int(len(yy_sorted[0])*0.2)-1] for el in yy_sorted],
        #                  [el[int(len(yy_sorted[0])*0.8)-1] for el in yy_sorted], color=self.color,
        #                  alpha=0.1, label=self.label)
        # plt.plot(xx, [el.mean() for el in yy_sorted], color=self.color)
        if not label:
            label = self.label
        if axis:
            if which == -1:
                for i in range(len(self.history_tables)):
                    if i == len(self.history_tables)-1:
                        self.plot_single_run(self.history_tables[i], y_feature, axis, label=label)
                    else:
                        self.plot_single_run(self.history_tables[i], y_feature, axis, label=None)
                axis.set_xlabel(" ".join(x_feature.split("_")))
                axis.set_ylabel(" ".join(y_feature.split("_")))
            else:
                self.plot_single_run(self.history_tables[which], y_feature, axis, label=label)
        else:
            xx = self.history_tables[0][x_feature]
            plt.plot(xx, self.history_tables[0][y_feature], label=label)
            plt.xlabel(" ".join(x_feature.split("_")))
            plt.ylabel(" ".join(y_feature.split("_")))
        if show:
            plt.grid()
            plt.legend()
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

    def plot_mean_feature(self, feature, condition=True, show=True, axis=None):
        if self.cell_tables:
            xx = sorted(self.cell_tables[0]["time_step"].unique())
            yy = self.cell_tables[0].groupby("time_step").aggregate(np.mean)
            if axis:
                axis.plot(xx, yy[feature])
                axis.set_ylabel(f"mean {feature}")
            else:
                plt.plot(xx, yy[feature])
                # plt.xlabel("time_step")
                plt.ylabel(f"mean {feature}")
            if show:
                plt.grid()
                plt.legend()
                plt.show()


class TSVVisualizator(Visualizator):
    def __init__(self,
                 root_path: str,
                 run_id: int,
                 color="blue", label="", read_cell_tables=False):
        super().__init__(root_path, run_id, color, label)
        files = [file.stem for file in Path(f"{root_path}/{run_id}").glob("*.tsv")]

        n_threads = max([int(file.split("_")[-1]) for file in files])
        self.history_tables = [pd.read_csv(f"{root_path}/{run_id}/history_{run_id}_{i}.tsv", sep="\t")
                               for i in range(1, n_threads+1)]
        if read_cell_tables:
            try:
                self.cell_tables = [pd.read_csv(f"{root_path}/{run_id}/cells_{run_id}_{i}.tsv", sep="\t")
                                    for i in range(1, n_threads+1)]
            except FileNotFoundError:
                self.cell_tables = []
        else:
            self.cell_tables = []
        try:
            self.genealogy_tables = [pd.read_csv(f"{root_path}/{run_id}/genealogy_{run_id}_{i}.tsv", sep="\t")
                                     for i in range(1, n_threads + 1)]
        except FileNotFoundError:
            self.genealogy_tables = []


class SQLVisualizator(Visualizator):
    def __init__(self,
                 root_path: str,
                 run_id,
                 color="blue", label="", read_cell_tables=False):
        super().__init__(root_path, run_id, color, label)
        self.folder = f"{root_path}/{run_id}"
        self.run_id = str(run_id).split("_")[0]
        files = [str(file) for file in Path(self.folder).glob("*.sqlite")]
        self.connections = [sqlite3.connect(file) for file in files]
        for con in self.connections:
            atexit.register(con.close)
        self.history_tables = [pd.read_sql_query("SELECT * FROM history", con) for con in self.connections]
        if read_cell_tables:
            try:
                self.cell_tables = [pd.read_sql_query("SELECT * FROM cells", con) for con in self.connections]
            except pd.io.sql.DatabaseError:  # no cells table
                self.cell_tables = []
        else:
            self.cell_tables = []
        # self.genealogy_tables = [pd.read_sql_query("SELECT * FROM genealogy", con) for con in self.connections]

        # If a run ends with a dead population that runs for many generations, it needs to be cropped.
        bs = [(table.n_cells > 0)[::-1] for table in self.history_tables]
        last_steps = [min(len(b), len(b) - np.argmax(b) - 1 + 1000) for b in bs]
        self.history_tables = [table.iloc[:last_steps[i], :] for i, table in enumerate(self.history_tables)]

    def plot_mean_feature(self, feature, condition=True, show=True, axis=None):
        super().plot_mean_feature(feature, condition, show, axis)

    def make_interactive_mode_figure(self, which=-1, show=False):
        _, ax = plt.subplots(3, 1)
        color = self.color
        self.color = "blue"
        self.plot("time_step", "n_cells", which, show=False, axis=ax[0])
        self.color = "grey"
        self.plot("time_step", "mean_damage", which, show=False, axis=ax[1])
        self.color = "green"
        self.plot("time_step", "mean_asymmetry", which, show=False, axis=ax[2], label="asymmetry")
        self.color = "red"
        self.plot("time_step", "mean_repair", which, show=False, axis=ax[2], label="repair")
        self.color = color
        if len(self.history_tables) == 1:
            self.plot_environment(ax[0], color="grey")
            self.plot_environment(ax[1], color="grey", y=-1)
            self.plot_environment(ax[2], color="grey", y=-0.01)

        ax[2].set_ylabel("mean asymmetry \n & repair")
        plt.legend()
        if show:
            plt.show()

    def plot_environment(self, axis, run=0, color=None, y=-8.0):
        if "environment" in self.history_tables[run].columns and \
                len(self.history_tables[run]["environment"].unique()) > 1:
            if color is None:
                color = self.color
            env = self.history_tables[run].environment
            i = 0
            start_x = 0
            while i < len(env):
                if not env[i]:
                    axis.plot([start_x, i], [y, y], linewidth=3, color=color)
                    while i < len(env) and not env[i]:
                        i += 1
                    start_x = i
                else:
                    i += 1
            axis.plot([start_x, i], [y, y], linewidth=3, color=color)

    def make_standard_plots(self, save_folder, which):
        """
        :param save_folder:
        :param which: run number or -1 if all the runs
        :return:
        """
        for feature, figure_name in zip(["n_cells",
                                         "mean_asymmetry",
                                         "mean_damage",
                                         "mean_repair"],
                                        ["population_size_dynamics",
                                         "asymmetry_dynamics",
                                         "damage_dynamics",
                                         "repair_dynamics"]):
            _, ax = plt.subplots()
            self.plot("time_step", feature, which=which, show=False, axis=ax)
            ax.set_title(" ".join([el.capitalize() for el in figure_name.split("_")]))
            self.plot_environment(ax, color="grey", y=0)
            plt.savefig(f"{save_folder}/{figure_name}.png")

    def make_figure_report(self, min_n_iterations=0):
        figures_folder = f"{self.folder}/{self.run_id}_figures"
        Path(figures_folder).mkdir(exist_ok=True)
        self.make_standard_plots(figures_folder, -1)
        plt.close()
        logging.info("Created standard plots")
        self.make_interactive_mode_figure()
        plt.savefig(f"{figures_folder}/interactive_mode_figure.png")
        plt.close()
        logging.info("Created interactive mode figure")
        if len(self.history_tables) > 1:
            individual_runs_path = Path(figures_folder)/Path("individual_runs")
            individual_runs_path.mkdir(exist_ok=True)
            for i in range(len(self.history_tables)):
                if len(self.history_tables[i].time_step) < min_n_iterations:
                    continue
                save_path = individual_runs_path/Path(str(i+1))
                save_path.mkdir(exist_ok=True)
                self.make_standard_plots(save_path, i)
                logging.info(f"Created standard plots for run {i}")
                self.make_interactive_mode_figure(i)
                plt.savefig(save_path/Path("interactive_mode_figure.png"))
                plt.close()
                logging.info(f"Created interactive mode figure for run {i}")

id_1 = 1669128159147082  # Mutation step = 0.05
id_2 = 1669128218000692  # Mutation step = 0.01
id_3 = 1669130957394782  # Mutation step = 0.05, mutation rate = 0.001
id_4 = 1669193798094970  # daec = 0.01
id_5 = 1669194873430621  # daec = 0, a=0
id_6 = 1669195448345880  # daec = 0, a=1
id_7 = 1669197993386483  # daec = 0, a=0.5
id_8 = 1669199773539028  # weird damage accumulation rate = 100
id_9 = 1669201571863388  # weird damage accumulation rate = 1000
id_10 = 1671481850523483  # story about asymmetry evolving in a population on the brink of extintion!
id_11 = 1671485681602655  # same, 10 runs, mutation rate = 0.02
id_12 = 1671660658135236  # asymmetry does not beat multiplicative repair
id_13 = 1671664065198253  # multiplicative repair
id_14 = 1671665041933698  # asymmetry is better than multiplicative repair?
id_15 = 1671726415351954  # additive repair

visualizator = SQLVisualizator(root_path=root_path,
                               run_id="1672268085464116_asymmetry_beats_multiplicative_repair",
                               color='blue')
visualizator.make_figure_report(min_n_iterations=500000)
# visualizator.make_interactive_mode_figure(show=True)


# Analysing population sizes with various asymmetry levels
def experiment_1():
    folders = [int(str(p).split("/")[-1]) for p in Path(root_path + "asymmetric_division_saves_population/").glob("*")]
    folders.sort()
    visualizators = [Visualizator(root_path=root_path+"asymmetric_division_saves_population/",
                                  run_id=folder) for folder in folders]
    visualizators.sort(key=lambda el: el.params["asymmetry"])

    asymmetries = [visualizator.params["asymmetry"] for visualizator in visualizators]
    print(asymmetries)
    stable_popsizes = [np.array([np.array(history_table.n_cells[:-100]).mean()
                                 for history_table in visualizator.history_tables])
                       for visualizator in visualizators]
    print(stable_popsizes)

    data = pd.DataFrame({asymmetry: popsize for asymmetry, popsize in zip(asymmetries, stable_popsizes)})
    data[asymmetries].plot(kind='box', title='Population size depending on the asymmetry level')
    plt.xticks(rotation=90)
    plt.xlabel("Asymmetry")
    plt.ylabel("Population size")
    plt.show()
