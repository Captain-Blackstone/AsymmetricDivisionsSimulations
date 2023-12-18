import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from MasterEquationPhageSimulation import PhageSimulation
from MasterEquationSimulationPCD import PCDSimulation


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
    parser.add_argument("--discretization_volume", type=int, default=41)
    parser.add_argument("--discretization_damage", type=int, default=1001)
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


def initialize_conditions_dictionary(simulationClass) -> dict:
    conditions = {"matrix": None,
                  "phi": None}
    if simulationClass in [PhageSimulation, PCDSimulation]:
        conditions["ksi"] = None
    return conditions


def check_all_asymmetries(repair: float,
                          a_steps: int,
                          params: dict,
                          path: str,
                          simulationClass,
                          conditions: dict,
                          a_min=0,
                          a_max=1,
                          **kwargs) -> (bool, np.array, float):

    parameters = params.copy()
    estimates_file = f"{path}/population_size_estimate.txt"
    equilibria = []
    for a in np.linspace(a_min, a_max, a_steps):
        # Do not rerun already existing estimations
        current_estimate = get_estimate(file=estimates_file, a_val=a, r_val=repair)
        if current_estimate is not None:
            equilibria.append(round(current_estimate))
            continue

        parameters["a"] = round(a, 10)
        parameters["r"] = round(repair, 10)
        simulation = simulationClass(params=parameters, save_path=path, **kwargs)

        # Initialize from previous state
        if any([el is None for el in conditions.values()]):
            simulation.run(1000000000000000000, save=False)
            for key in conditions.keys():
                conditions[key] = getattr(simulation, key)
            simulation = simulationClass(params=parameters, save_path=path, **kwargs)
        if all([el is not None for el in conditions.values()]) and conditions["matrix"].sum() > 0:
            if conditions["matrix"].sum() < 1:
                conditions["matrix"] = conditions["matrix"] / conditions["matrix"].sum()
            if conditions.get("ksi") is not None and conditions["ksi"] < 1:
                conditions["ksi"] = 10000
            for key, val in conditions.items():
                setattr(simulation, key, val)

        # Run the simulation
        logging.info(f"starting simulation with params: {parameters}")
        simulation.run(1000000000000000000)
        for key in conditions.keys():
            conditions[key] = getattr(simulation, key)

        # Add equilibrium population size for this value of asymmetry to the list
        if simulation.convergence_estimate is not None:
            equilibria.append(round(simulation.convergence_estimate))

    # If for given value of repair asymmetry is neutral, stop scanning, we know the rest of the landscape
    a_neutral = len(set(equilibria)) == 1 and len(equilibria) == a_steps and equilibria[0] > 1
    return a_neutral, conditions


def scan_grid(params: dict,
              r_steps: int,
              a_steps: int,
              path: str,
              simulationClass,
              max_r=None,
              a_min=0,
              a_max=1,
              **kwargs):
    if max_r is None:
        max_r = min(params["D"], params["E"]) \
            if params.get("D") is not None and params["D"] != 0 \
            else (min(params["F"] / 100, params["E"]))
        if simulationClass is PCDSimulation:
            influx_estimate = params["C"]*kwargs["phage_influx"]/kwargs["discretization_damage"]
            possible_upper_bounds = [influx_estimate, params["F"], params["E"]]
            possible_upper_bounds = list(filter(lambda el: el > 0, possible_upper_bounds))
            max_r = min(possible_upper_bounds)
    a_neutral = False
    conditions = initialize_conditions_dictionary(simulationClass)
    for r in np.linspace(0, max_r, r_steps):
        a_neutral, conditions = check_all_asymmetries(repair=r,
                                                      a_steps=a_steps,
                                                      params=params,
                                                      path=path,
                                                      simulationClass=simulationClass,
                                                      conditions=conditions,
                                                      a_min=a_min,
                                                      a_max=a_max,
                                                      **kwargs)
    return a_neutral


def scan_until_death_or_a_neutral(params: dict,
                                  path: str,
                                  a_steps: int,
                                  a_neutral: bool,
                                  simulationClass,
                                  **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    rr = sorted(df[1].unique())
    conditions = initialize_conditions_dictionary(simulationClass)
    if len(rr) > 1 and not a_neutral:
        r_step = rr[1] - rr[0]
        r = max(df[1])
        if len(df.loc[df[1] == r]) < a_steps:
            a_neutral, conditions = check_all_asymmetries(repair=r,
                                                          a_steps=a_steps,
                                                          path=path,
                                                          simulationClass=simulationClass,
                                                          conditions=conditions,
                                                          params=params, **kwargs)

        # While the populations with maximum checked repair survive with at least some degree of asymmetry
        while len(df.loc[(df[1] == max(df[1])) & (df[2] > 1)]) > 0:
            r_step *= 2
            r = min(r + r_step, params["E"])
            a_neutral, conditions = check_all_asymmetries(repair=r,
                                                          a_steps=a_steps,
                                                          path=path,
                                                          simulationClass=simulationClass,
                                                          conditions=conditions,
                                                          params=params, **kwargs)
            if a_neutral:
                print("a neutral, breaking. max_r: ", r)
                break
            df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
            if r == params["E"]:
                print("reached maximum r=E, breaking, r=", r)
                break


def find_the_peak(params: dict, path: str, a_steps: int, simulationClass, **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    for a in [0, 1]:
        a_1 = df.loc[df[0] == a].drop_duplicates()
        rr = np.array(list(a_1.sort_values(1)[1]))
        popsizes = np.array(list(a_1.sort_values(1)[2]))
        if all(np.ediff1d(popsizes) < 0):  # The peak is r = 0
            min_r = rr[0]
            max_r = rr[1]
            mag1 = mag2 = popsizes[1] - popsizes[0]
        else:  # The peak is r != 0
            if len(np.ediff1d(popsizes)[np.ediff1d(popsizes) > 0]) == 0:
                continue
            mag1, mag2 = (np.ediff1d(popsizes)[np.ediff1d(popsizes) > 0][-1],
                          np.ediff1d(popsizes)[np.ediff1d(popsizes) < 0][0])
            min_r = rr[:-1][np.ediff1d(popsizes) > 0][-1]
            if min_r == rr[-2]:
                max_r = rr[-1]
            else:
                max_r = rr[list(rr).index(min_r) + 2]
        iteration = 0
        conditions = initialize_conditions_dictionary(simulationClass)
        while abs(mag1) > 1 or abs(mag2) > 1:
            iteration += 1
            for r in np.linspace(min_r, max_r, 4):
                _, conditions = check_all_asymmetries(repair=r,
                                                      a_steps=a_steps,
                                                      path=path,
                                                      simulationClass=simulationClass,
                                                      conditions=conditions,
                                                      params=params, **kwargs)

            df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
            a_1 = df.loc[df[0] == 1].drop_duplicates()
            rr = np.array(list(a_1.sort_values(1)[1]))
            popsizes = np.array(list(a_1.sort_values(1)[2]))
            if all(np.ediff1d(popsizes) < 0):
                min_r = rr[0]
                max_r = rr[1]
                mag1 = mag2 = popsizes[1] - popsizes[0]
            else:
                mag1, mag2 = (np.ediff1d(popsizes)[np.ediff1d(popsizes) > 0][-1],
                              np.ediff1d(popsizes)[np.ediff1d(popsizes) < 0][0])
                min_r = rr[:-1][np.ediff1d(popsizes) > 0][-1]
                if min_r == rr[-2]:
                    max_r = rr[-1]
                else:
                    max_r = rr[list(rr).index(min_r) + 2]
            if iteration > 20:
                break
