from MasterEquationPhageSimulation import PhageSimulation
from command_line_interface_functions import *
import atexit
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.WARNING)


def check_all_asymmetries(repair: float,
                          a_steps: int,
                          params: dict,
                          path: str,
                          starting_matrix=None,
                          starting_phi=None,
                          starting_ksi=None,
                          **kwargs) -> (bool, np.array, float):
    parameters = params.copy()
    if path != "":
        estimates_file = f"{path}/population_size_estimate.txt"
    equilibria = []
    matrix, phi, ksi = starting_matrix, starting_phi, starting_ksi
    for a in np.linspace(0, 1, a_steps):

        # Do not rerun already existing estimations
        if path != "":
            current_estimate = get_estimate(file=estimates_file, a_val=a, r_val=repair)
            if current_estimate is not None:
                equilibria.append(round(current_estimate))
                continue

        parameters["a"] = round(a, 5)
        parameters["r"] = repair
        simulation = PhageSimulation(params=parameters, save_path=path, **kwargs)
        # Initialize from previous state
        if matrix is not None and phi is not None and matrix.sum() > 0 and ksi is not None:
            if matrix.sum() < 1:
                matrix = matrix / matrix.sum()
            simulation.matrix = matrix
            simulation.phi = phi
            simulation.ksi = ksi
        # Run the simulation
        logging.info(f"starting simulation with params: {parameters}")
        matrix, phi, ksi = simulation.run(1000000000000000000, save=True if path != "" else False)

        # Add equilibrium population size for this value of asymmetry to the list
        if simulation.convergence_estimate is not None:
            equilibria.append(round(simulation.convergence_estimate))

    # If for given value of repair asymmetry is neutral, stop scanning, we know the rest of the landscape
    a_neutral = len(set(equilibria)) == 1 and len(equilibria) == a_steps and equilibria[0] > 1
    death = all([el < 1 for el in equilibria])
    return a_neutral, death, matrix, phi, ksi


def guess_max_r(params: dict, repair_steps: int, death: bool, **kwargs):
    if params["D"] != 0:
        r_bound = min(params["E"], params["D"]*1.1)
    else:
        max_r_guesses = [params["E"], params["E"] / (repair_steps/2)]
        stop_guessing = False
        test_parameters = params.copy()
        dead_guesses = []
        while not stop_guessing:
            print("trying", max_r_guesses[-1])
            a_neutral, death_with_current_r, _, _, _, = check_all_asymmetries(repair=max_r_guesses[-1], a_steps=2,
                                                                           params=test_parameters,
                                                                           path="",
                                                                           starting_matrix=None,
                                                                           starting_phi=None,
                                                                              starting_ksi=None,
                                                                           **kwargs)
            if a_neutral:
                max_r_guesses.append(max_r_guesses[-1] / (repair_steps/2))
                if death and death_with_current_r:
                    dead_guesses.append(max_r_guesses)
            else:
                stop_guessing = True
            if death and len(dead_guesses) > 50:
                break
            logging.info("Tried so far: " + str(max_r_guesses[:-1]))
        r_bound = max_r_guesses[-1]
        print("choosing ", r_bound, "as max r")
    return r_bound


def scan_grid(params: dict,
              r_steps: int,
              a_steps: int,
              path: str,
              max_r=None,
              **kwargs):
    if max_r is None:
        max_r = min(params["D"], params["E"]) if params.get("D") is not None and params["D"] != 0 else (
            min(params["F"] / 100, params["E"]))
    a_neutral = False
    matrix, phi, ksi = None, None, None
    for r in np.linspace(0, max_r, r_steps)[1:]:
        a_neutral, death, matrix, phi, ksi = check_all_asymmetries(repair=r,
                                                              a_steps=a_steps,
                                                              params=params,
                                                              path=path,
                                                              starting_matrix=matrix,
                                                              starting_phi=phi,
                                                              starting_ksi=ksi,
                                                              **kwargs)
        if a_neutral:
            break
    return a_neutral


def find_the_peak(params: dict, path: str, a_steps: int, **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    a_1 = df.loc[df[0] == 1].drop_duplicates()
    rr = np.array(list(a_1.sort_values(1)[1]))
    popsizes = np.array(list(a_1.sort_values(1)[2]))
    # The peak is r = 0
    if all(np.ediff1d(popsizes) < 0):
        return
    if len(np.ediff1d(popsizes)[np.ediff1d(popsizes) > 0]) == 0:
        return
    mag1, mag2 = np.ediff1d(popsizes)[np.ediff1d(popsizes) > 0][-1], np.ediff1d(popsizes)[np.ediff1d(popsizes) < 0][0]
    min_r = rr[:-1][np.ediff1d(popsizes) > 0][-1]
    max_r = rr[list(rr).index(min_r) + 2]

    while abs(mag1) > 1 or abs(mag2) > 1:
        matrix, phi, ksi = None, None, None
        for r in np.linspace(min_r, max_r, 3):
            print(r)
            a_neutral, matrix, phi, ksi = check_all_asymmetries(repair=r,
                                                           a_steps=a_steps,
                                                           path=path,
                                                           starting_matrix=matrix,
                                                           starting_phi=phi,
                                                                starting_ksi=ksi,
                                                           params=params, **kwargs)

        df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
        a_1 = df.loc[df[0] == 1].drop_duplicates()
        rr = np.array(list(a_1.sort_values(1)[1]))
        popsizes = np.array(list(a_1.sort_values(1)[2]))
        # The peak is r = 0
        if all(np.ediff1d(popsizes) < 0):
            return

        mag1, mag2 = np.ediff1d(popsizes)[np.ediff1d(popsizes) > 0][-1], np.ediff1d(popsizes)[np.ediff1d(popsizes) < 0][
            0]
        min_r = rr[:-1][np.ediff1d(popsizes) > 0][-1]
        max_r = rr[list(rr).index(min_r) + 2]


def scan_until_death_or_a_neutral(params: dict, path: str, a_steps: int, a_neutral: bool, **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    rr = sorted(df[1].unique())
    if len(rr) > 1 and not a_neutral:
        r_step = rr[1] - rr[0]
        r = max(df[1])
        matrix, phi, ksi = None, None, None
        if len(df.loc[df[1] == r]) < a_steps:
            a_neutral, death, matrix, phi, ksi = check_all_asymmetries(repair=r,
                                                          a_steps=a_steps,
                                                          path=path,
                                                          starting_matrix=matrix,
                                                          starting_phi=phi,
                                                                starting_ksi=ksi,
                                                          params=params, **kwargs)

        # While the populations with maximum checked repair survive with at least some degree of asymmetry
        while len(df.loc[(df[1] == max(df[1])) & (df[2] > 1)]) > 0:
            r_step *= 2
            r = min(r + r_step, params["E"])
            a_neutral, death, matrix, phi, ksi = check_all_asymmetries(repair=r,
                                                                a_steps=a_steps,
                                                                path=path,
                                                                starting_matrix=matrix,
                                                                starting_phi=phi,
                                                                starting_ksi=ksi,
                                                                params=params, **kwargs)
            if a_neutral:
                print("a neutral, breaking. max_r: ", r)
                break
            df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
            if r == params["E"]:
                print("reached maximum r=E, breaking, r=", r)
                break

if __name__ == "__main__":
    np.random.seed(10)
    parser = argparse.ArgumentParser(prog="MasterEquation simulator")
    tune_parser(parser)
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)

    if args.mode in ["local", "interactive"]:
        save_path = f"../../data/master_equation/" \
                    f"{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}"
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}"

    Path(save_path).mkdir(exist_ok=True)
    atexit.register(lambda: write_completion(save_path))
    params = {"A": args.A, "B": args.B, "C": args.C, "D": args.D, "E": args.E, "F": args.F, "G": args.G}
    logging.info("Checking if phage dies out at r = 0")
    a_neutral_at_r_0, death, _, _, _ = check_all_asymmetries(repair=0,
                                                          a_steps=args.a,
                                                          params=params,
                                                          path=save_path,
                                                          starting_matrix=None,
                                                          starting_phi=None,
                                                          mode=args.mode,
                                                          discretization_volume=args.discretization_volume,
                                                          discretization_damage=args.discretization_damage)
    if a_neutral_at_r_0:
        max_r = args.E
    else:
        max_r = guess_max_r(params=params, death=death, mode=args.mode, repair_steps=args.r,
                            discretization_volume=args.discretization_volume,
                            discretization_damage=args.discretization_damage)
    a_neutral = scan_grid(params=params,
                          r_steps=args.r, a_steps=args.a,
                          path=save_path,
                          max_r=max_r,
                          mode=args.mode,
                          discretization_volume=args.discretization_volume,
                          discretization_damage=args.discretization_damage)

    scan_until_death_or_a_neutral(params=params,
                                  a_neutral=a_neutral,
                                  path=save_path,
                                  mode=args.mode,
                                  a_steps=args.a,
                                  discretization_volume=args.discretization_volume,
                                  discretization_damage=args.discretization_damage)
    find_the_peak(params=params,
                  path=save_path,
                  mode=args.mode,
                  a_steps=args.a,
                  discretization_volume=args.discretization_volume,
                  discretization_damage=args.discretization_damage)
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")
