import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from Selection_landscape import Landscape

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


class SmallGillespie:
    def __init__(self,
                 landscape_obj: Landscape,
                 starting_state: str):
        self.landscape = landscape_obj
        self.current_state = starting_state

    def get_rates(self):
        probs = np.array([self.landscape.graph[el[0]][el[1]]["weight"]
                          for el in self.landscape.graph.edges(self.current_state)])
        population_sizes = self.landscape.population_sizes[(ASYMMETRIES[int(self.current_state.split("_")[0])],
                                                            REPAIRS[int(self.current_state.split("_")[1])])]

        waiting_times = np.array([
            self.landscape.fixation_times[(ASYMMETRIES[int(el[0].split("_")[0])], REPAIRS[int(el[0].split("_")[1])],
                                           ASYMMETRIES[int(el[1].split("_")[0])], REPAIRS[int(el[1].split("_")[1])])]
            for el in self.landscape.graph.edges(self.current_state)])
        rates = probs * population_sizes / waiting_times
        return rates

    def change_state(self, i):
        self.current_state = list(self.landscape.graph.edges(self.current_state))[i][1]


class Gillespie:
    def __init__(self, landscape_obj: Landscape, starting_state: str, n_instances: int):
        self.starting_state = starting_state
        self.instances = [SmallGillespie(landscape_obj, starting_state) for _ in range(n_instances)]
        self.history = []
        self.im = None
        self.int_matrix = None
        self.hh_t = None

    def step(self):
        rates = []
        events = []
        for i, instance in enumerate(self.instances):
            rr = instance.get_rates()
            rates.extend(rr)
            for j in range(len(rr)):
                events.append([i, j, rr[j]])
        event = events[np.random.choice(list(range(len(events))), p=np.array(rates) / sum(rates))]
        prev_state = self.instances[event[0]].current_state
        self.instances[event[0]].change_state(event[1])
        self.history.append([event[0],
                             prev_state,
                             self.instances[event[0]].current_state,
                             np.random.poisson(1 / sum([el[2] for el in events]))])

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def get_normalized_history(self):
        mn = min([el[3] for el in self.history])
        return [[el[0], el[1], el[2], int(el[3] / mn * 2)] for el in self.history]

    def draw(self, save_path: str):
        fig = plt.figure()
        ax = plt.axes(xlim=(-0.5, 10.5), ylim=(-0.5, 10.5))
        ax.set_xticks(range(11))
        ax.set_xticklabels(ASYMMETRIES)
        ax.set_yticks(range(11))
        ax.set_yticklabels(REPAIRS)
        ax.set_xlabel("asymmetry")
        ax.set_ylabel("repair")
        a = np.random.random((11, 11))
        self.im = plt.imshow(a, interpolation='none')
        self.int_matrix = np.zeros((11, 11))
        start_1, start_2 = list(map(int, self.starting_state.split("_")))
        self.int_matrix[start_1, start_2] = len(self.instances)
        nh = self.get_normalized_history()
        self.hh_t = []
        for item in nh:
            self.hh_t.append(item[1:3])
            for _ in range(item[3] - 1):
                self.hh_t.append(["0_0", "0_0"])
        print(len(self.hh_t), "frames")
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init_func,
                                       frames=list(range(len(self.hh_t))), interval=1, blit=True)
        print('saving')
        writer = animation.FFMpegWriter(fps=200, bitrate=1800)
        anim.save(save_path, writer=writer)

    def init_func(self):
        self.im.set_data(np.zeros((11, 11)).astype(int))
        return [self.im]

    def animate(self, i):
        minus, plus = self.hh_t[i]
        minus = list(map(int, minus.split("_")))
        plus = list(map(int, plus.split("_")))
        self.int_matrix[minus[1], minus[0]] -= 1
        self.int_matrix[plus[1], plus[0]] += 1
        self.im.set_data((self.int_matrix / self.int_matrix.sum()) ** 0.3)
        return [self.im]


if __name__ == "__main__":
    from simulations.scripts.selection_landscapes_scripts.Selection_landscape import dataframes
    from pathlib import Path
    dataframe_path = dataframes["linear_da_lower_cost"]
    landscape = Landscape(dataframe_path)
    g = Gillespie(landscape_obj=landscape, starting_state="0_0", n_instances=100)
    g.run(n_steps=7000)
    g.draw(save_path=f"{str(Path(dataframe_path).parent)}/evolutionary_trajectory_gillespie.mp4")
