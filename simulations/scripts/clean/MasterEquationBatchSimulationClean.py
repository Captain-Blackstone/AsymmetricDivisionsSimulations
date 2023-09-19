from MasterEquationSimulation import Simulation
from command_line_interface_functions import *
import atexit
import pandas as pd
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.WARNING)


def scan_until_death_or_a_neutral(params: dict, path: str, a_steps: int, a_neutral: bool, **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    rr = sorted(df[1].unique())
    if len(rr) > 1 and not a_neutral:
        r_step = rr[1] - rr[0]
        r = max(df[1])
        matrix, phi = None, None
        if len(df.loc[df[1] == r]) < a_steps:
            a_neutral, matrix, phi, check_all_asymmetries(repair=r,
                                                          a_steps=a_steps,
                                                          path=path,
                                                          simulationClass=Simulation,
                                                          starting_matrix=matrix,
                                                          starting_phi=phi,
                                                          params=params, **kwargs)

        # While the populations with maximum checked repair survive with at least some degree of asymmetry
        while len(df.loc[(df[1] == max(df[1])) & (df[2] > 1)]) > 0:
            r_step *= 2
            r = min(r + r_step, params["E"])
            a_neutral, matrix, phi = check_all_asymmetries(repair=r,
                                                           a_steps=a_steps,
                                                           path=path,
                                                           simulationClass=Simulation,
                                                           starting_matrix=matrix,
                                                           starting_phi=phi,
                                                           params=params, **kwargs)
            if a_neutral:
                print("a neutral, breaking. max_r: ", r)
                break
            df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
            if r == params["E"]:
                print("reached maximum r=E, breaking, r=", r)
                break


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
        matrix, phi = None, None
        for r in np.linspace(min_r, max_r, 3):
            print(r)
            a_neutral, matrix, phi = check_all_asymmetries(repair=r,
                                                           a_steps=a_steps,
                                                           path=path,
                                                           simulationClass=Simulation,
                                                           starting_matrix=matrix,
                                                           starting_phi=phi,
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

        print(mag1, mag2)


if __name__ == "__main__":
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
    a_neutral = scan_grid(params={"A": args.A, "B": args.B, "C": args.C,
                                  "D": args.D, "E": args.E, "F": args.F, "G": args.G},
                          r_steps=args.r, a_steps=args.a,
                          path=save_path,
                          simulationClass=Simulation,
                          mode=args.mode,
                          discretization_volume=args.discretization_volume,
                          discretization_damage=args.discretization_damage)
    scan_until_death_or_a_neutral(params={"A": args.A, "B": args.B, "C": args.C,
                                  "D": args.D, "E": args.E, "F": args.F, "G": args.G},
                                  a_neutral=a_neutral,
                                  path=save_path,
                                  mode=args.mode,
                                  a_steps=args.a,
                                  discretization_volume=args.discretization_volume,
                                  discretization_damage=args.discretization_damage)
    find_the_peak(params={"A": args.A, "B": args.B, "C": args.C,
                          "D": args.D, "E": args.E, "F": args.F, "G": args.G},
                          path=save_path,
                          mode=args.mode,
                          a_steps=args.a,
                          discretization_volume=args.discretization_volume,
                          discretization_damage=args.discretization_damage)
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")



