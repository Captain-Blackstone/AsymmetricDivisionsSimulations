import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from MasterEquationPhageSimulation import PhageSimulation
from MasterEquationSimulationPCD import PCDSimulation


def tune_parser(parser: argparse.ArgumentParser, ar_type=int):
    parser.add_argument("-m", "--mode", default="local", type=str, choices=["cluster", "local", "interactive"], help="mode=cluster outputs no information to the command line and stores the outputs in the ./data/ folder; mode=local outputs information about number of iterations the simulation run through and stores the output in the ../../data/ folder; mode=interactive opens an interactive window where the user can set all the parameters and observe the change in variable values while the simulation is running. It is even possible to change the parameters in the middle of the run, but the simulation has to be paused while the parameters are tweaked.")
    parser.add_argument("-ni", "--niterations", default=100000, type=int, help="number of iterations of a simulation to carry out. It is advisable to set it to an arbitrary large number - the simulation will stop when equilibrium is achieved or when the time limit (20 minutes) is reached")
    parser.add_argument("--run_name", type=str, default="", help="not used")
    parser.add_argument("-A", type=float, default=28.0, help="A parameter, growth rate (see the paper")  # nu * x * phi0
    parser.add_argument("-B", type=float, help="B parameter, dilution rate (see the paper)")  # R / V
    parser.add_argument("-C", type=float, help="C parameter, nutrient and phage acquizition rate (see the paper)")  # nu * x / V
    parser.add_argument("-D", type=float, default=0, help="rate of linear phage replication rate (was not included in the paper, equals to 0 by default")  # d / K ?
    parser.add_argument("-E", type=float, help="E parameter, cost of immunity (see the paper)")  # 0 < E <= 1
    parser.add_argument("-F", type=float, help="F parameter, phage replication rate (see the paper)")  # 0 <= F
    parser.add_argument("-G", type=float, default=1.0, help="not used")  # G
    parser.add_argument("-H", type=float, default=0, help="not used")
    parser.add_argument("-a", type=ar_type, help="a parameter, investment in PCD, between 0 and 1 (see the paper")  # 0 <= a <= 1
    parser.add_argument("-r", type=ar_type, help="r parameter, investment in immunity, between 0 and E (see the paper)")  # 0 <= r <= E
    parser.add_argument("--discretization_volume", type=int, default=41, help="number of cell size bins")
    parser.add_argument("--discretization_damage", type=int, default=1001, help="maximum conceivable number of phages (T=1 means burst size is equal to this parameter)")
    parser.add_argument("--save_path", type=str, default=None, help="path to the folder where the result of the simulation will be stored")
    parser.add_argument("--debug", action='store_true', help="whether to output information about the simulation to the command line")


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
    logging.info(f"running a={a_val}, r={r_val}")
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
                          a_min=0.0,
                          a_max=1.0,
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


def check_all_repairs(asymmetry: float,
                      r_steps: int,
                      params: dict,
                      path: str,
                      simulationClass,
                      conditions: dict,
                      r_min=0,
                      r_max=None,
                      **kwargs) -> (bool, np.array, float):
    if r_max is None:
        r_max = params["E"]
    parameters = params.copy()
    estimates_file = f"{path}/population_size_estimate.txt"
    equilibria = []
    for r in np.linspace(r_min, r_max, r_steps):
        # Do not rerun already existing estimations
        current_estimate = get_estimate(file=estimates_file, a_val=asymmetry, r_val=r)
        if current_estimate is not None:
            equilibria.append(round(current_estimate))
            continue

        parameters["a"] = round(asymmetry, 10)
        parameters["r"] = round(r, 10)
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


def scan_grid_log(params: dict,
                  r_steps: int,
                  a_steps: int,
                  path: str,
                  simulationClass,
                  aa=None,
                  rr=None,
                  **kwargs):
    parameters = params.copy()
    estimates_file = f"{path}/population_size_estimate.txt"
    if aa is None:
        aa = [1] + list(1 - np.logspace(0, 2, a_steps) / 100)
    if rr is None:
        # rr = [0] + list(np.logspace(0, 2, r_steps-1) / 100 * params["E"])
        rr = [0] + list(np.logspace(0, 2, r_steps) / 100 * params["E"])
    for r in rr:
        conditions = initialize_conditions_dictionary(simulationClass)
        for a in aa:
            # Do not rerun already existing estimations
            current_estimate = get_estimate(file=estimates_file, a_val=a, r_val=r)
            if current_estimate is not None:
                continue

            parameters["a"] = round(a, 10)
            parameters["r"] = round(r, 10)

            if (any([el is None for el in conditions.values()]) or
                    (conditions["matrix"] is not None and conditions["matrix"].sum() < 1)):
                simulation = simulationClass(params=parameters, save_path=path, **kwargs)
                simulation.run(1000000000000000000, save=False)
                for key in conditions.keys():
                    conditions[key] = getattr(simulation, key)

            if all([el is not None for el in conditions.values()]) and conditions["matrix"].sum() > 0:
                # Don't start from dead population
                if conditions["matrix"].sum() < 1:
                    conditions["matrix"] = conditions["matrix"] / conditions["matrix"].sum()
                # If no phage in chemostat, add some phage
                if conditions.get("ksi") is not None and conditions["ksi"] < 1:
                    conditions["ksi"] = 10000

            logging.info(f"starting simulation with params: {parameters}")
            # Create simulation
            simulation = simulationClass(params=parameters, save_path=path, **kwargs)

            # Initialize with conditions
            for key, val in conditions.items():
                setattr(simulation, key, val)

            # Run simulation
            simulation.run(1000000000000000000)

            # Save the final conditions
            for key in conditions.keys():
                conditions[key] = getattr(simulation, key)
        break

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


def find_the_peak_pcd(params: dict, path: str, a_steps: int, simulationClass, **kwargs):
    for _ in range(20):
        df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
        df.columns = ["a", "r", "population_size", "converged", "type", "ksi", "estimated_equilibrium_damage", "burst_size"]
        df = df.loc[df.type != "undefined"]
        current_peak = df.loc[df.population_size == df.population_size.max()]
        if current_peak.population_size.values[0] < 1:
            return
        r_peak = current_peak.r.values[0]
        a_peak = current_peak.a.values[0]
        a_vals, r_vals = [], []
        for trait_peak, trait, not_trait, trait_list in zip([a_peak, r_peak],
                                                            ["a", "r"], ["r", "a"],
                                                            [a_vals, r_vals]):
            # sorted_unique = sorted(list(df.loc[df.not_trait == current_peak[not_trait].values[0]][trait].unique()))
            sorted_unique = sorted(list(df[trait].unique()))
            index = sorted_unique.index(trait_peak)
            neighbour_indices = [index - 1, index + 1]  # list(filter(lambda el: 0 <= el <= len(sorted_unique) - 1, [index - 1, index + 1]))
            left = sorted_unique[neighbour_indices[0]] if neighbour_indices[0] >= 0 else 0
            if neighbour_indices[1] <= len(sorted_unique) - 1:
                right = sorted_unique[neighbour_indices[1]]
            elif trait == "r":
                right = params["E"]
            elif trait == "a":
                right = 1
            try:
                left_delta = current_peak.population_size.values[0] - df.loc[df[trait] == left].population_size.values[0]
            except IndexError:
                left_delta = 1000
            try:
                right_delta = current_peak.population_size.values[0] - df.loc[df[trait] == right].population_size.values[0]
            except IndexError:
                right_delta = 1000
            if left_delta < 10 and right_delta < 10:
                trait_list.extend([trait_peak])
            else:
                trait_list.extend(list(np.linspace(left, right, 4).round(10)))
        if len(set(a_vals)) == 1 and len(set(r_vals)) == 1:
            return
        scan_grid_log(params=params,
                      r_steps=2,
                      a_steps=2,
                      path=path,
                      simulationClass=simulationClass,
                      aa=a_vals,
                      rr=r_vals,
                      **kwargs)


def get_landscape_contour(params: dict, path: str, simulationClass, **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    df.columns = ["a", "r", "population_size", "converged", "type", "ksi", "estimated_equilibrium_damage", "burst_size"]
    df = df.loc[df.type != "undefined"]
    current_peak = df.loc[df.population_size == df.population_size.max()]
    if current_peak.population_size.values[0] < 1:
        return
    r_peak = current_peak.r.values[0]
    a_peak = current_peak.a.values[0]
    conditions = initialize_conditions_dictionary(simulationClass)
    check_all_asymmetries(repair=r_peak, a_steps=21, params=params,
                          path=path, simulationClass=simulationClass, conditions=conditions, **kwargs,
                          a_min=1-params["T"], a_max=1)
    conditions = initialize_conditions_dictionary(simulationClass)
    check_all_repairs(asymmetry=a_peak, r_steps=21, params=params,
                      path=path, simulationClass=simulationClass, conditions=conditions, **kwargs)
    check_all_repairs(asymmetry=1-params["T"], r_steps=21, params=params,
                      path=path, simulationClass=simulationClass, conditions=conditions, **kwargs)


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
