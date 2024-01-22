from MasterEquationSimulationPCD import PCDSimulation
from command_line_interface_functions import *
import atexit
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator PCD")
    tune_parser(parser)
    parser.add_argument("--nondivision_threshold", type=int, default=0)
    parser.add_argument("--phage_influx", type=float, default=0)
    parser.add_argument("--refine", type=float, default=0)
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)

    if args.mode in ["local", "interactive"]:
        save_path = f"../../data/master_equation/" \
                    f"{args.A}_{args.B}_{args.C}_{args.phage_influx}_{args.E}_{args.F}"
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.phage_influx}_{args.E}_{args.F}"
    Path(save_path).mkdir(exist_ok=True)
    atexit.register(lambda: write_completion(save_path))
    if args.refine != 0:
        max_r = None
        if Path(f"{save_path}/population_size_estimate.txt").exists():
            estimates = pd.read_csv(f"{save_path}/population_size_estimate.txt", sep=",", header=None)
            max_r = estimates.loc[estimates[2] == estimates[2].max()][1].values[0]
        a_neutral = scan_grid(params={"A": args.A, "B": args.B, "C": args.C,
                                      "D": args.D, "E": args.E, "F": args.F, "G": args.G},
                              r_steps=args.r, a_steps=args.a,
                              path=save_path,
                              simulationClass=PCDSimulation,
                              mode=args.mode,
                              discretization_volume=args.discretization_volume,
                              discretization_damage=args.discretization_damage,
                              nondivision_threshold=args.nondivision_threshold,
                              phage_influx=args.phage_influx,
                              a_min=args.refine,
                              a_max=1,
                              max_r=max_r)

    else:

        scan_grid_log(params={"A": args.A, "B": args.B, "C": args.C,
                              "D": args.D, "E": args.E, "F": args.F, "G": args.G},
                      r_steps=args.r, a_steps=args.a,
                      path=save_path,
                      simulationClass=PCDSimulation,
                      mode=args.mode,
                      discretization_volume=args.discretization_volume,
                      discretization_damage=args.discretization_damage,
                      nondivision_threshold=args.nondivision_threshold,
                      phage_influx=args.phage_influx)

        find_the_peak_pcd(params={"A": args.A, "B": args.B, "C": args.C,
                                  "D": args.D, "E": args.E, "F": args.F, "G": args.G},
                          path=save_path,
                          mode=args.mode,
                          a_steps=args.a,
                          simulationClass=PCDSimulation,
                          discretization_volume=args.discretization_volume,
                          discretization_damage=args.discretization_damage,
                          nondivision_threshold=args.nondivision_threshold,
                          phage_influx=args.phage_influx
                          )
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")
