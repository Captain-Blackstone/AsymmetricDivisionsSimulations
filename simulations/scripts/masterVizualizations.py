from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmin, argrelmax
from tqdm import tqdm


class Visualizer:
    def __init__(self, folder):
        self.folder = f"../data/master_equation/{folder}"
        self.simulation_results = []
        for folder in Path(self.folder).glob("*"):
            s = SimulationResults(folder)
            if s.times is not None:
                self.simulation_results.append(s)

    def plot(self, show=False):
        max_t = 0
        for res in self.simulation_results:
            smoothed, t_smoothed = res.get_smoothed()
            time = max(t_smoothed)
            max_t = max(time, max_t)
            plt.plot(t_smoothed, smoothed)
        plt.gca().set_prop_cycle(None)
        eqs = []
        times = []
        for res in self.simulation_results:
            eq, w = res.get_equilibrium_estimate()
            if eq is not None:
                eqs.append(eq)
                times.append(w)
                plt.plot([0, max_t], [eq, eq], lw=w/10)
        eq = (np.array(eqs) * np.array(times)/sum(times)).sum()
        print(eq)
        plt.plot([0, max_t], [eq, eq], lw=3, color="grey", linestyle="--")
        eq = np.array(eqs).mean()
        print(eq)
        plt.plot([0, max_t], [eq, eq], lw=3, color="black", linestyle="--")
        if show:
            plt.show()

    def plot_equilibrium_estimate(self, i, show=False):
        simulation = self.simulation_results[i]
        estimates, times = [], []
        for j in tqdm(range(0, len(simulation.times), 500)):
            estimate, peaks = simulation.get_equilibrium_estimate(j)
            if estimate is not None:
                estimates.append(estimate)
                times.append(simulation.real_times[j])
        plt.plot(times, estimates)
        # plt.plot(simulation.times, simulation.population_sizes)
        if show:
            plt.show()


class SimulationResults:
    def __init__(self, folder):
        history_file = Path(f"{folder}/population_size_history.txt")
        if history_file.exists():
            with history_file.open("r") as fl:
                self.times = list(map(float, fl.readline().strip().split()))
                self.population_sizes = list(map(float, fl.readline().strip().split()))
                self.real_times = list(map(float, fl.readline().strip().split()))
        else:
            self.times, self.population_sizes, self.real_times = None, None, None

    def get_smoothed(self, popsizes=None, times=None):
        if popsizes is None:
            popsizes = np.array(self.population_sizes)
        if times is None:
            times = np.array(self.times)
        minima, t_minima = popsizes[argrelmin(popsizes, order=10)], times[argrelmin(popsizes, order=10)]
        maxima, t_maxima = popsizes[argrelmax(popsizes, order=10)], times[argrelmax(popsizes, order=10)]
        minima, maxima, t_minima, t_maxima = minima[:min(len(minima), len(maxima))], \
            maxima[:min(len(minima), len(maxima))], \
            t_minima[:min(len(minima), len(maxima))], \
            t_maxima[:min(len(minima), len(maxima))]
        smoothed, t_smoothed = (minima + maxima) / 2, (t_minima + t_maxima) / 2
        return smoothed, t_smoothed

    def get_equilibrium_estimate(self, i):
        smoothed, t_smoothed = self.get_smoothed(np.array(self.population_sizes)[:i], np.array(self.times)[:i])
        mins, maxes = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
        if len(mins) and len(maxes):
            return (mins[-1] + maxes[-1])/2, len(mins) + len(maxes)
        else:
            return None, None

visualizer = Visualizer("1685131265447297_knowledgeable_kingfish")
for i in range(len(visualizer.simulation_results)):
    visualizer.plot_equilibrium_estimate(i, show=False)
plt.show()
# visualizer = Visualizer("1685126354110857long_run")
# visualizer.plot_equilibrium_estimate(0, show=True)
