import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd


def tune_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-m", "--mode", default="local", type=str, choices=["cluster", "local", "interactive"])
    parser.add_argument("-ni", "--niterations", default=100000, type=int)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("-A", type=float)  # nu * x * phi0
    parser.add_argument("-B", type=float)  # R / V
    parser.add_argument("-C", type=float)  # nu * x / V
    parser.add_argument("-D", type=float)  # d / K ?
    parser.add_argument("-E", type=float)  # 0 < E <= 1
    parser.add_argument("-F", type=float)  # 0 <= F
    parser.add_argument("-G", type=float)  # G
    parser.add_argument("-H", type=float, default=0)
    parser.add_argument("-a", type=int)  # 0 <= a <= 1
    parser.add_argument("-r", type=int)  # 0 <= r <= E
    parser.add_argument("--discretization_volume", type=int, default=251)
    parser.add_argument("--discretization_damage", type=int, default=251)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", action='store_true')


def write_completion(path):
    with open(f"{path}/scanning.txt", "a") as fl:
        fl.write("complete\n")


def get_estimate(file: str, a_val: float, r_val: float):
    if Path(file).exists():
        print(str(file))
        estimates = pd.read_csv(file, sep=",", header=None)
        relevant_estimates = estimates.loc[(abs(estimates[0] - a_val) < 1e-10) & (abs(estimates[1] - r_val) < 1e-10), :]
        if len(relevant_estimates) > 0:
            logging.info(f"skipping a={a_val}, r={r_val}, estimate already exists")
            return list(relevant_estimates[2])[0]
    return None


def check_all_asymmetries(repair: float,
                          a_steps: int,
                          params: dict,
                          path: str,
                          simulationClass,
                          starting_matrix=None,
                          starting_phi=None,
                          **kwargs) -> (bool, np.array, float):
    parameters = params.copy()
    estimates_file = f"{path}/population_size_estimate.txt"
    equilibria = []
    matrix, phi = starting_matrix, starting_phi
    for a in np.linspace(0, 1, a_steps):

        # Do not rerun already existing estimations
        current_estimate = get_estimate(file=estimates_file, a_val=a, r_val=repair)
        if current_estimate is not None:
            equilibria.append(round(current_estimate))
            continue

        parameters["a"] = round(a, 10)
        parameters["r"] = round(repair, 10)
        simulation = simulationClass(params=parameters, save_path=path, **kwargs)

        # Initialize from previous state
        if matrix is None and phi is None:
            matrix, phi = simulation.run(1000000000000000000, save=False)
            simulation = simulationClass(params=parameters, save_path=path, **kwargs)
        if matrix is not None and phi is not None and matrix.sum() > 0:
            if matrix.sum() < 1:
                matrix = matrix / matrix.sum()
            simulation.matrix = matrix
            simulation.phi = phi

        # Run the simulation
        logging.info(f"starting simulation with params: {parameters}")
        matrix, phi = simulation.run(1000000000000000000)

        # Add equilibrium population size for this value of asymmetry to the list
        if simulation.convergence_estimate is not None:
            equilibria.append(round(simulation.convergence_estimate))

    # If for given value of repair asymmetry is neutral, stop scanning, we know the rest of the landscape
    a_neutral = len(set(equilibria)) == 1 and len(equilibria) == a_steps and equilibria[0] > 1
    return a_neutral, matrix, phi


def scan_grid(params: dict,
              r_steps: int,
              a_steps: int,
              path: str,
              simulationClass,
              max_r=None,
              **kwargs):
    if max_r is None:
        max_r = min(params["D"], params["E"]) if params.get("D") is not None and params["D"] != 0 else (
            min(params["F"] / 100, params["E"]))
    a_neutral = False
    matrix, phi = None, None
    for r in np.linspace(0, max_r, r_steps):
        a_neutral, matrix, phi = check_all_asymmetries(repair=r,
                                                       a_steps=a_steps,
                                                       params=params,
                                                       path=path,
                                                       simulationClass=simulationClass,
                                                       starting_matrix=matrix,
                                                       starting_phi=phi,
                                                       **kwargs)
        if a_neutral:
            break
    return a_neutral
