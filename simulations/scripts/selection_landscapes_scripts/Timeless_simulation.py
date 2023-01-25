import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
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


class SmallTimelessSimulation:
    def __init__(self, landscape_obj: Landscape, starting_state: str):
        self.landscape = landscape_obj
        self.current_node = starting_state
        self.positions = []

    def run(self, n_steps=2000):
        for i in range(n_steps):
            pos = list(map(int, self.current_node.split("_")))
            self.positions.append(pos)
            probs = [self.landscape.graph[el[0]][el[1]]["weight"]
                     for el in self.landscape.graph.edges(self.current_node)]
            if np.random.uniform() < sum(probs):
                self.current_node = np.random.choice(a=[el[1] for el in self.landscape.graph.edges(self.current_node)],
                                                     p=np.array(probs) / sum(probs))


class TimelessSimulation:
    def __init__(self, landscape_obj: Landscape, starting_state: str, n_instances: int):
        self.instances = [SmallTimelessSimulation(landscape_obj, starting_state) for _ in range(n_instances)]

    def run(self, n_steps) -> None:
        for instance in self.instances:
            instance.run(n_steps)

    def draw_adaptive_peak(self, save_path: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        mtx = np.zeros((11, 11))
        end_positions = [walk.positions[-1] for walk in self.instances]
        for e in end_positions:
            mtx[10 - e[1], e[0]] += 1
        mtx /= mtx.sum()
        mtx = mtx ** 0.3
        plt.imshow(mtx)
        ax.set_xticks(range(11))
        ax.set_xticklabels(ASYMMETRIES)
        ax.set_yticks(range(11))
        ax.set_yticklabels(REPAIRS[::-1])
        ax.set_xlabel("asymmetry")
        ax.set_ylabel("repair")
        plt.savefig(save_path)

    def walk_visualization_video(self, save_path):
        fig = plt.figure()
        ax = plt.axes(xlim=(-0.5, 10.5), ylim=(-0.5, 10.5))
        ax.set_xticks(range(11))
        ax.set_xticklabels(ASYMMETRIES)
        ax.set_yticks(range(11))
        ax.set_yticklabels(REPAIRS)
        ax.set_xlabel("asymmetry")
        ax.set_ylabel("repair")

        a = np.random.random((11, 11))
        im = plt.imshow(a, interpolation='none')

        def init_func():
            im.set_data(np.zeros((11, 11)).astype(int))
            return [im]

        def animate(i):
            points = [walk.positions[i] for walk in self.instances]
            mtx = np.zeros((11, 11))
            for point in points:
                mtx[point[1], point[0]] += 1
            mtx /= mtx.sum()
            mtx = mtx ** 0.3
            im.set_data(mtx)
            return [im]

        anim = animation.FuncAnimation(fig, animate, init_func=init_func,
                                       frames=len(self.instances[0].positions), interval=20, blit=True)

        writer = animation.FFMpegWriter(fps=60, bitrate=1800)
        anim.save(save_path, writer=writer)


if __name__ == "__main__":
    from Selection_landscape import dataframes
    from pathlib import Path
    dataframe_path = dataframes["linear_da_lower_cost"]
    landscape = Landscape(dataframe_path)
    tls = TimelessSimulation(landscape_obj=landscape, starting_state="0_0", n_instances=100)
    tls.run(2000)
    tls.draw_adaptive_peak(save_path=f"{str(Path(dataframe_path).parent)}/peak.png")
    tls.walk_visualization_video(save_path=f"{str(Path(dataframe_path).parent)}/evolutionary_trajectory.mp4")
