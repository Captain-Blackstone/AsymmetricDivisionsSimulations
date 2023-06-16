import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
from pathlib import Path
import logging
import argparse
from copy import copy


class Simulation:
    def __init__(self,
                 discretization_volume: int,
                 discretization_damage: int,
                 params: dict,
                 save_path: str,
                 mode: str):
        self.p = np.linspace(1, 2, discretization_volume)
        self.q = np.linspace(0, 1, discretization_damage)
        rhos = np.outer(1 / self.p, self.q)
        self.damage_death_rate = (rhos / (1 - rhos)) ** params["G"]
        self.damage_death_rate[np.isinf(self.damage_death_rate)] = \
            self.damage_death_rate[~np.isinf(self.damage_death_rate)].max()
        self.p = self.p[:-1]
        self.damage_death_rate = self.damage_death_rate[:-1, :]

        self.params = params
        self.history = History(simulation_obj=self, save_path=save_path)

        self.mode = mode

        self.population_size_estimate = None
        self.population_structure_estimate = None
        self.phi_estimate = None
        self.error = None
        self.delta_estimation = None
        self.converged = False
        self.lowest_eigenvalues = dict()

    @staticmethod
    def find_nearest(array: np.array, value: float) -> float:
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def get_derivative_matrix(self, phi):

        # Growth
        growth_mtx = np.zeros([len(self.p) * len(self.q)] * 2)
        growth_rate = self.params["A"] * (1 - self.params['r'] / self.params["E"]) * phi
        diagonal_array = np.array(list(itertools.repeat(list(self.p), len(self.q)))).flatten()
        growth_mtx -= np.diag(diagonal_array) * growth_rate

        subdiagonal_array = copy(diagonal_array)
        zero_out = [False] * len(self.p)
        zero_out[-1] = True
        zero_out *= len(self.q)
        subdiagonal_array[np.array(zero_out)] = 0

        subdiagonal = np.concatenate([np.zeros_like(diagonal_array).reshape(1, len(diagonal_array)),
                                      np.diag(subdiagonal_array)[:-1, :]])

        growth_mtx += subdiagonal * growth_rate

        # Division
        division_mtx = np.zeros([len(self.p) * len(self.q)] * 2)
        damage = np.arange(len(self.q))
        where_to_divide_1 = damage * (1 - self.params["a"]) / 2
        where_to_divide_1 = np.array([int(el) for el in where_to_divide_1])
        where_to_divide_2 = damage - where_to_divide_1

        division = np.zeros_like(division_mtx)
        ii = np.arange(len(self.p) * len(self.q))

        for where_to_divide in [where_to_divide_1, where_to_divide_2]:
            for index in range(len(where_to_divide)):
                division[ii[ii % len(self.p) == 0][where_to_divide[index]],
                ii[(ii + 1) % len(self.p) == 0][index]] += growth_rate * self.p[-1]
        division_mtx += division

        # Death
        death_mtx = np.zeros([len(self.p) * len(self.q)] * 2)
        death_mtx -= np.diag(self.damage_death_rate.T.flatten() + self.params["B"])

        # Damage accumulation
        da_mtx = np.zeros([len(self.p) * len(self.q)] * 2)
        da_mtx -= np.diag(diagonal_array * self.params["D"] + np.outer(self.q, np.ones_like(self.p)).flatten() *
                          self.params["F"])
        da_mtx += np.concatenate([np.zeros((len(self.p), len(self.p) * len(self.q))),
                                  np.diag(diagonal_array * self.params["D"] +
                                          np.outer(self.q, np.ones_like(self.p)).flatten() *
                                          self.params["F"])[:-len(self.p), :]])

        # Repair
        r_mtx = np.zeros([len(self.p) * len(self.q)] * 2)
        repair_array = copy(diagonal_array)
        repair_array[:len(self.p)] = 0
        r_mtx -= np.diag(repair_array * self.params["r"])
        r_mtx += np.concatenate([np.diag(diagonal_array * self.params["r"])[len(self.p):, :],
                                 np.zeros((len(self.p), len(self.p) * len(self.q)))])

        return growth_mtx + division_mtx + death_mtx + da_mtx + r_mtx

    def get_equilibrium_population_size(self, max_time: float):
        reshaped_p = np.array(list(itertools.repeat(self.p, len(self.q)))).reshape([len(self.p) * len(self.q), 1])
        delta_estimation = 1000
        min_phi, max_phi = 0, 1
        n_steps = 11
        while not self.converged:
            lowest_vals = []
            lowest_vecs = []
            phis = np.linspace(min_phi, max_phi, n_steps)
            step = phis[1] - phis[0]
            if self.mode == "local":
                iterator = tqdm(phis)
            else:
                iterator = phis
            logging.debug(" ".join(map(str, phis)))
            for phi in iterator:
                A = self.get_derivative_matrix(phi)
                vals, vecs = np.linalg.eig(A)
                least_val = self.find_nearest(np.array(vals), 0)

                vec = vecs[:, vals == least_val]
                if vec.shape[-1] != 1:
                    vec = vec[:, 0]
                    vec = vec.reshape([len(vec), 1])
                lowest_vals.append(least_val)
                lowest_vecs.append(vec)
                self.lowest_eigenvalues[phi] = least_val
            if all([el < 0 for el in lowest_vals]) or all([el > 0 for el in lowest_vals]):
                if n_steps == 101:
                    break
                else:
                    n_steps = (n_steps - 1) * 10 + 1
                    continue
            n_steps = 11
            mask = np.array(lowest_vals) == self.find_nearest(np.array(lowest_vals), 0)
            phi, vector = phis[mask][0], np.array(lowest_vecs)[mask][0]
            vector = np.real(vector)
            if all(vector < 0):
                vector *= -1
            vector[vector < 0] = 0
            k = (1 - phi) * self.params["B"] / (self.params["C"] * phi) / (reshaped_p * vector).sum()
            vector = vector * k
            A = self.get_derivative_matrix(phi)
            error = ((A @ (vector.reshape([len(self.p) * len(self.q), 1]))) ** 2).sum()
            min_phi, max_phi = max(phi - step / 2, 0), min(phi + step / 2, 1)
            if self.population_size_estimate is not None:
                delta_estimation = abs(vector.sum() - self.population_size_estimate)
            self.converged = (delta_estimation < 1) and self.phi_estimate != phi
            self.population_size_estimate = vector.sum()
            self.population_structure_estimate = vector
            self.phi_estimate = phi
            self.delta_estimation = delta_estimation
            self.error = error
            self.history.record()
            # plt.scatter(phis, lowest_vals)
            # plt.show()
            if time.time() - self.history.starting_time > max_time:
                break
            logging.info(f"phi estimate: {self.phi_estimate}, "
                         f"lowest eigenvalue: {min(lowest_vals)} "
                         f"population size estimate: {self.population_size_estimate}, "
                         f"delta estimation: {self.delta_estimation}, "
                         f"error: {self.error}")
        self.history.save()


class History:
    def __init__(self, simulation_obj: Simulation, save_path: str):
        self.simulation = simulation_obj
        self.population_sizes = []
        self.phis = []
        self.errors = []
        self.delta_estimates = []

        self.save_path = save_path
        Path(self.save_path).mkdir(exist_ok=True)
        self.starting_time = time.time()

    def record(self) -> None:
        self.population_sizes.append(self.simulation.population_size_estimate)
        self.phis.append(self.simulation.phi_estimate)
        self.errors.append(self.simulation.error)
        self.delta_estimates.append(self.simulation.delta_estimation)

    def save(self):
        with open(f"{self.save_path}/history.txt", "w") as fl:
            fl.write(" ".join(list(map(str, self.phis))) + '\n')
            fl.write(" ".join(list(map(str, self.population_sizes))) + '\n')
            fl.write(" ".join(list(map(str, self.delta_estimates))) + '\n')
            fl.write(" ".join(list(map(str, self.errors))) + '\n')

        with open(f"{self.save_path}/meta.txt", "w") as fl:
            fl.write(f"p array size: {len(self.simulation.p)}\n")
            fl.write(f"q array size: {len(self.simulation.q)}\n")
            fl.write("time: " + str((time.time() - self.starting_time)/60) + '\n')
            fl.write("converged: " + str(self.simulation.converged) + '\n')

        with open(f"{self.save_path}/population_structure.txt", "w") as fl:
            if self.simulation.population_structure_estimate is not None:
                fl.write(" ".join(list(map(str, self.simulation.population_structure_estimate.flatten()))) + '\n')
            else:
                fl.write('\n')

        with open(f"lowest_eigenvalues.txt", "w") as fl:
            fl.write(" ".join(list(map(str, self.simulation.lowest_eigenvalues.keys()))) + '\n')
            fl.write(" ".join(list(map(str, self.simulation.lowest_eigenvalues.values()))) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation solver")
    parser.add_argument("-m", "--mode", default="local", type=str, choices=["cluster", "local"])
    parser.add_argument("-mt", "--max_time", default=600.0, type=float)
    parser.add_argument("-A", type=float)  # nu * x * phi0
    parser.add_argument("-B", type=float)  # R / V
    parser.add_argument("-C", type=float)  # nu * x / V
    parser.add_argument("-D", type=float)  # d / K ?
    parser.add_argument("-E", type=float)  # 0 < E <= 1
    parser.add_argument("-F", type=float)  # 0 <= F
    parser.add_argument("-G", type=float)  # G
    parser.add_argument("-a", type=float)  # 0 <= a <= 1
    parser.add_argument("-r", type=float)  # 0 <= r <= E
    parser.add_argument("--discretization_volume", type=int, default=251)
    parser.add_argument("--discretization_damage", type=int, default=251)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if args.mode == "local":
        from tqdm import tqdm
        save_path = f"../data/master_equation_solver/" \
                    f"{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}_{args.a}_{args.r}"
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}_{args.a}_{args.r}"
    parameters = {"A": args.A, "B": args.B, "C": args.C, "D": args.D*args.discretization_damage,
                  "E": args.E*args.discretization_damage, "F": args.F, "G": args.G,
                  "a": args.a, "r": args.r*args.discretization_damage}
    Path(save_path).mkdir(exist_ok=True)
    logging.info("Params: ", " ".join(["=".join([key, val]) for key, val in parameters.items()]))
    simulation = Simulation(params=parameters,
                            save_path=save_path if args.save_path is None else args.save_path,
                            discretization_volume=args.discretization_volume,
                            discretization_damage=args.discretization_damage,
                            mode=args.mode)

    simulation.get_equilibrium_population_size(args.max_time)
