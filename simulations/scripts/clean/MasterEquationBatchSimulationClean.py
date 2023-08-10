from MasterEquationSimulation import Simulation
from command_line_interface_functions import *
import atexit
import pandas as pd
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.WARNING)


def scan_until_death_or_a_neutral(params: dict, path: str, a_neutral: bool, **kwargs):
    df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)
    rr = sorted(df[1].unique())
    if len(rr) > 1 and not a_neutral:
        r_step = rr[1] - rr[0]
        r = max(df[1])
        # While the populations with maximum checked repair survive with at least some degree of asymmetry
        while len(df.loc[(df[1] == max(df[1])) & (df[2] > 1)]) > 0:
            r_step *= 2
            r = min(r + r_step, args.E)
            a_neutral = check_all_asymmetries(repair=r,
                                              a_steps=args.a,
                                              path=path,
                                              simulationClass=Simulation,
                                              params=params, **kwargs)
            if a_neutral:
                break
            df = pd.read_csv(f"{path}/population_size_estimate.txt", header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator")
    tune_parser(parser)
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG)

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
                                  discretization_volume=args.discretization_volume,
                                  discretization_damage=args.discretization_damage)
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")



