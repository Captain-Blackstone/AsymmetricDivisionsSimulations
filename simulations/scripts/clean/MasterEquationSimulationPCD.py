import time

from MasterEquationPhageSimulation import PhageSimulation
from master_equation_pcd_functions import divide, accumulate_phage


class PCDSimulation(PhageSimulation):
    def __init__(self, params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251,
                 nondivision_threshold: int = 1,
                 phage_influx: float = 0,
                 ):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage, phage_influx)
        self.nondivision_threshold = nondivision_threshold

    def divide(self):
        return divide(matrix=self.proposed_new_matrix, q=self.q, nondivision_threshold=self.nondivision_threshold)

    def accumulate_damage(self):
        return accumulate_phage(matrix=self.matrix,
                                C=self.params["C"], F=self.params["F"],
                                ksi=self.ksi,
                                delta_t=self.delta_t,
                                p=self.p, q=self.q)

    def upkeep_after_step(self):
        super().upkeep_after_step()
        self.matrix[self.rhos > 1 - self.params["a"]] = 0
        time.sleep(0.0001)


if __name__ == "__main__":
    import atexit
    import logging
    import argparse
    from command_line_interface_functions import tune_parser, write_completion
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="MasterEquation simulator PCD")
    tune_parser(parser, ar_type=float)
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
    simulation = PCDSimulation(mode=args.mode,
                               params={"A": args.A, "B": args.B, "C": args.C, "D": 0,
                                       "E": args.E, "F": args.F, "G": args.G, "a": args.a, "r": args.r},
                               save_path=save_path,
                               discretization_volume=args.discretization_volume,
                               discretization_damage=args.discretization_damage,
                               nondivision_threshold=args.nondivision_threshold,
                               phage_influx=args.phage_influx)
    if args.mode == "interactive":
        simulation.run_interactive()
    else:
        simulation.run(10000000000, save=False)


