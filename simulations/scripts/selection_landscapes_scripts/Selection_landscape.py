import pandas as pd
import numpy as np
import networkx as nx
from colour import Color
import matplotlib.pyplot as plt
from pathlib import Path

dataframes = {
    "linear_da":
        "../../selection_coefficients/linear_da/linear_da_selection_coefficients_raw_data.tsv",
    "linear_da_lower_cost":
        "../../selection_coefficients/linear_da_lower_cost/linear_da_lower_cost_selection_coefficients_raw_data.tsv"
}

params = {
    "asymmetry": {
        "min": 0,
        "max": 1,
        "mutation_step": 0.1
    },
    "repair": {
        "min": 0,
        "max": 1e-6,
        "mutation_step": 1e-7
    }
}

ASYMMETRIES = np.linspace(params["asymmetry"]["min"],
                          params["asymmetry"]["max"],
                          int((params["asymmetry"]["max"]-params["asymmetry"]["min"]) /
                              params["asymmetry"]["mutation_step"]) + 1).round(5)
REPAIRS = np.linspace(params["repair"]["min"],
                      params["repair"]["max"],
                      int((params["repair"]["max"]-params["asymmetry"]["min"]) /
                          params["repair"]["mutation_step"]) + 1).round(10)


class Landscape:
    def __init__(self, df_path):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path, sep="\t")
        self.population_sizes = self.get_population_sizes_from_df()
        self.probs_of_fixation = self.get_fixation_probability_dict()
        self.fixation_times = self.get_fixation_times()
        self.graph = self.create_graph()

    def get_population_sizes_from_df(self) -> pd.DataFrame:
        population_size_df = self.df.loc[(self.df.population_start != "None") &
                                         (self.df.population_start.notna())].copy()
        population_size_df.loc["population_start"] = population_size_df.population_start.astype(int)
        return population_size_df.groupby(["wt_a", "wt_r"]).aggregate("population_start").mean()

    def get_fixation_probability_dict(self):
        resulting_dict = {}
        for i, a in enumerate(ASYMMETRIES):
            for j, r in enumerate(REPAIRS):
                target_cell = [i, j]
                neighbors = [[i+1, j], [i-1, j], [i, j+1], [i, j-1]]
                for n in neighbors:
                    if n[0] > len(ASYMMETRIES)-1 or n[1] > len(REPAIRS)-1 or n[0] < 0 or n[1] < 0:
                        continue
                    wt_a, mut_a = ASYMMETRIES[target_cell[0]], ASYMMETRIES[n[0]]
                    wt_r, mut_r = REPAIRS[target_cell[1]], REPAIRS[n[1]]
                    wt_win = len(self.df.loc[(self.df.wt_a == wt_a) &
                                             (self.df.wt_r == wt_r) &
                                             (self.df.mut_a == mut_a) &
                                             (self.df.mut_r == mut_r) & (self.df.winner == "wt")])
                    mut_win = len(self.df.loc[(self.df.wt_a == wt_a) &
                                              (self.df.wt_r == wt_r) &
                                              (self.df.mut_a == mut_a) &
                                              (self.df.mut_r == mut_r) & (self.df.winner == "mut")])
                    if wt_win + mut_win > 0:
                        s = mut_win / (wt_win + mut_win)
                    else:
                        s = None
                    resulting_dict[f"{target_cell[0]}_{target_cell[1]}_vs_{n[0]}_{n[1]}"] = s
        return resulting_dict

    def create_graph(self) -> nx.DiGraph:
        result_graph = nx.DiGraph()
        for i, a in enumerate(ASYMMETRIES):
            for j, r in enumerate(REPAIRS):
                result_graph.add_node(f"{i}_{j}")
        for key, val in self.probs_of_fixation.items():
            if val is None:
                continue
            result_graph.add_edge(*key.split("_vs_"), weight=val)
        return result_graph

    def get_fixation_times(self):
        time_df = self.df.loc[(self.df.winner == "mut")]
        time_df = time_df.groupby(["wt_a", "wt_r", "mut_a", "mut_r"]).aggregate("n_steps").mean()
        for wt_a in ASYMMETRIES:
            for wt_r in REPAIRS:
                for mut_a in ASYMMETRIES:
                    for mut_r in REPAIRS:
                        if (wt_a, wt_r, mut_a, mut_r) not in time_df.index:
                            time_df[(wt_a, wt_r, mut_a, mut_r)] = 1e100
        return time_df

    def draw_graph(self):
        fig, ax = plt.subplots(figsize=(20, 10))

        # Get the color array
        colors = list(Color("red").range_to(Color("green"), 50))
        space = []
        for u, v in self.graph.edges:
            weight1 = self.graph[u][v]['weight']
            try:
                weight2 = self.graph[v][u]['weight']
            except KeyError:
                continue
            space.append(weight1/(weight1+weight2))
        bounds = np.linspace(0.1, 0.9, 50)
        resulting_colors = []
        for u, v in self.graph.edges:
            weight1 = self.graph[u][v]['weight']
            try:
                weight2 = self.graph[v][u]['weight']
            except KeyError:
                resulting_colors.append("red")
                continue
            current_color = colors[0]
            for color, bound in zip(colors, bounds):
                if weight1/(weight1+weight2) >= bound:
                    current_color = color
                else:
                    break
            resulting_colors.append(current_color)

        # Get node sizes list
        sizes = []
        for i, a in enumerate(ASYMMETRIES):
            for j, r in enumerate(REPAIRS):
                if (a, r) in self.population_sizes.index:
                    sizes.append(self.population_sizes[(a, r)])
                else:
                    sizes.append(0)

        nx.draw(self.graph,
                pos={node: (int(node.split("_")[0]), int(node.split("_")[1])) for node in self.graph.nodes},
                width=[(self.graph[u][v]['weight'])*5 for u, v in self.graph.edges],
                connectionstyle="arc3,rad=0.1",
                edge_color=[str(el) for el in colors], node_size=[s*10 for s in sizes], ax=ax)
        ax.set_xlabel("asymmetry", fontsize=15)
        ax.set_ylabel("repair", fontsize=15)
        ax.set_xticks(range(11))
        ax.set_xticklabels(ASYMMETRIES)
        ax.set_yticks(range(11))
        ax.set_yticklabels(REPAIRS)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_axis_on()
        plt.savefig(f"{str(Path(self.df_path).parent)}/landscape.png")


if __name__ == "__main__":
    landscape = Landscape(df_path=dataframes["linear_da_lower_cost"])
    landscape.draw_graph()
    plt.show()
