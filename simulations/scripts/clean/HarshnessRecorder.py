from HarshnessTest import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Harshness Recorder simulator")
    tune_parser(parser)
    args = parser.parse_args()
    if args.mode in ["local"]:
        from tqdm import tqdm

        save_path = f"../data/master_equation/" \
                    f"{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}"
        if args.mode == "interactive":
            from master_interactive_mode_clean import Drawer
    else:
        if args.mode == "interactive":
            from master_interactive_mode_clean import Drawer
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.D}_{args.E}_{args.F}_{args.G}"

    Path(save_path).mkdir(exist_ok=True)
    parameters = {"A": args.A, "B": args.B, "C": args.C,
                  "D": args.D, "E": args.E, "F": args.F,
                  "G": args.G, "a": 0, "r": 0}
    simulation = HarhnessSimulation(params=parameters, mode=args.mode,
                                    save_path=str(save_path) if args.save_path is None else args.save_path,
                                    discretization_volume=args.discretization_volume,
                                    discretization_damage=args.discretization_damage)
    simulation.run(10000000000)
    target_harshness = simulation.convergence_estimate
    with open(f"{save_path}/harshness.csv", "w") as fl:
        fl.write(f"{args.B},{args.D},{args.E},{args.F},{target_harshness}\n")