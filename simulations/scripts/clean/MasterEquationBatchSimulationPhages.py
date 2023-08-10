from MasterEquationPhageSimulation import PhageSimulation
from command_line_interface_functions import *
import atexit
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def guess_max_r(params: dict):
    max_r_guesses = [min(params["F"] / 100, params["E"])]
    stop_guessing = False
    test_parameters = params.copy()
    while not stop_guessing:
        test_parameters["r"] = max_r_guesses[-1]
        convergence_estimates = dict()
        for asymmetry in [0, 1]:
            test_parameters["a"] = asymmetry
            simulation = PhageSimulation(params=test_parameters, mode=args.mode,
                                         save_path=str(save_path) if args.save_path is None else args.save_path,
                                         discretization_volume=args.discretization_volume,
                                         discretization_damage=args.discretization_damage)
            simulation.matrix /= simulation.matrix.sum()
            simulation.phi = 1
            simulation.ksi = 1
            logging.info(f"running simulation with params {simulation.params}")
            simulation.run(args.niterations, save=False)
            convergence_estimates[asymmetry] = simulation.convergence_estimate
        if int(convergence_estimates[0]) != int(convergence_estimates[1]):
            stop_guessing = True
        else:
            max_r_guesses.append(max_r_guesses[-1] / args.r)
        logging.info("Tried so far: " + str(max_r_guesses))
    print("max_r_guesses: ", max_r_guesses)
    print("length: ", len(max_r_guesses))
    if len(max_r_guesses) > 1:
        r_bound = max_r_guesses[-2]
        print("choosing second to last: ", r_bound)
    else:
        r_bound = max_r_guesses[-1]
        print("choosing last: ", r_bound)
    return r_bound


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator")
    tune_parser(parser, phage=True)
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG)

    if args.mode in ["local", "interactive"]:
        save_path = f"../../data/master_equation/" \
                    f"{args.A}_{args.B}_{args.C}_{args.E}_{args.F}_{args.G}"
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.E}_{args.F}_{args.G}"

    Path(save_path).mkdir(exist_ok=True)
    atexit.register(lambda: write_completion(save_path))
    params = {"A": args.A, "B": args.B, "C": args.C, "E": args.E, "F": args.F, "G": args.G}
    logging.info("Checking if phage dies out at r = 0")
    a_neutral_at_r_0, _, _ = check_all_asymmetries(repair=0,
                                                   a_steps=args.a,
                                                   params=params,
                                                   path=save_path,
                                                   simulationClass=PhageSimulation,
                                                   starting_matrix=None,
                                                   starting_phi=None,
                                                   mode=args.mode,
                                                   discretization_volume=args.discretization_volume,
                                                   discretization_damage=args.discretization_damage)
    if a_neutral_at_r_0:
        max_r = args.E
    else:
        max_r = guess_max_r(params=params)
    print("My r bound: ", max_r)
    a_neutral = scan_grid(params=params,
                          r_steps=args.r, a_steps=args.a,
                          path=save_path,
                          simulationClass=PhageSimulation,
                          max_r=max_r,
                          mode=args.mode,
                          discretization_volume=args.discretization_volume,
                          discretization_damage=args.discretization_damage)
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")
