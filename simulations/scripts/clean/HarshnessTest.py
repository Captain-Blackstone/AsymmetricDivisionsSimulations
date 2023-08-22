import numpy as np
from scipy.signal import argrelmin, argrelmax
import logging
import argparse

from pathlib import Path
from sklearn.linear_model import LinearRegression
from convergence_functions import (get_peaks, convergence, equilibrium_N)
from master_equation_functions import update_nutrient, grow, repair_damage, divide
from MasterEquationSimulation import Simulation, History
from master_equation_functions import clear_nonexistent
from command_line_interface_functions import tune_parser



def death(matrix: np.array, damage_death_rate: np.array, B: float, delta_t: float) -> np.array:
    those_that_die_from_dilution = B * delta_t * matrix
    those_that_die_from_damage = damage_death_rate * delta_t * matrix
    dead = those_that_die_from_dilution + those_that_die_from_damage
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] < 1e-50 and dead[i, j] > matrix[i, j]:
                dead[i, j] = matrix[i, j]
    return dead, those_that_die_from_damage.sum(), those_that_die_from_dilution.sum()



logging.basicConfig(level=logging.WARNING)


def gaussian_2d(x, y, mean_x, mean_y, var_x, var_y):
    return np.exp(-(np.power(x - mean_x, 2) / (2 * var_x) + np.power(y - mean_y, 2) / (2 * var_y)))


class HarhnessSimulation(Simulation):
    def __init__(self,
                 params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage)
        self.history = HarshnessHistory(simulation_obj=self, save_path=save_path)
        self.harshness = 0

    def equilibrium_N(self, peaks):
        return equilibrium_N(peaks)

    def check_convergence_v2(self):
        critical_period = self.max_delta_t * 10000
        logging.info(f"checking convergence, critical period - {critical_period}")
        # Claiming convergence only if critical period of time passed
        if self.history.times[-1] > critical_period:
            logging.info("really checking convergence")
            ii = (-np.array(self.history.times) + self.history.times[-1]) < critical_period
            if len(set(np.round(np.array(self.history.harshnesses)[ii], 1))) == 1 and len(
                    np.round(np.array(self.history.harshnesses)[ii], 1)) > 1:
                # Last 'critical period' of time was with the same population size
                self.converged = True
                self.convergence_estimate = self.harshness
                logging.info(f"same population size for {critical_period} time")
            else:
                peaks = get_peaks(self.history.harshnesses)
                if convergence(peaks) == "cycle":
                    self.converged = True
                    self.convergence_estimate = self.equilibrium_N(peaks)
                    logging.info("got a cycle")
                minima, maxima, t_minima, t_maxima = self.history.get_peaks()
                minima, maxima, t_minima, t_maxima = minima[-min(len(minima), len(maxima)):], \
                    maxima[-min(len(minima), len(maxima)):], \
                    t_minima[-min(len(minima), len(maxima)):], \
                    t_maxima[-min(len(minima), len(maxima)):]
                if len(minima) >= 2 and len(maxima) >= 2:  # If there were more than two minima and maxima
                    logging.info("convergence estimate could change now")
                    estimate = (minima[-1] + maxima[-1]) / 2  # Estimate based on last two 1st order peaks
                    # if self.convergence_estimate_first_order is not None:
                    #       print("prev n of peaks", len(minima) + len(maxima), 'current n of peaks', self.convergence_estimate_first_order[1])
                    if self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period / 4 and \
                            len(minima) + len(maxima) != self.convergence_estimate_first_order[1] and \
                            round(self.convergence_estimate_first_order[0], 2) == round(estimate, 1):
                        self.converged = True
                        self.convergence_estimate = self.convergence_estimate_first_order[0]
                        logging.info(
                            f"converged, same 1st order convergence estimate {estimate} as before: "
                            f"{self.convergence_estimate_first_order}")
                    # Else if there was no 1st order convergence estimate or
                    # there is one and some additional peaks arrived, update the 1st order convergence estimate
                    elif self.convergence_estimate_first_order is None or \
                            self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period / 4 and \
                            len(minima) + len(maxima) != self.convergence_estimate_first_order[1]:
                        self.convergence_estimate_first_order = [estimate, len(minima) + len(maxima), self.time]
                        logging.info(
                            f"changing 1st order convergence estimate: {self.convergence_estimate_first_order}")
                smoothed, t_smoothed = (minima + maxima) / 2, (t_minima + t_maxima) / 2
                if len(smoothed) > 5:
                    index_array = np.where(np.round(smoothed, 1) != np.round(smoothed, 1)[-1])[0]
                    if len(index_array) == 0:
                        last_time = t_smoothed[0]
                    else:
                        last_time = t_smoothed[np.max(index_array) + 1]
                    if self.history.times[-1] - last_time > critical_period:
                        self.converged = True
                        self.convergence_estimate = self.harshness
                        logging.info(f"converged, same population size for {critical_period} time")
                smoothed_minima, smoothed_maxima = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
                if len(smoothed_minima) >= 2 and len(smoothed_maxima) >= 2:
                    logging.info("convergence estimate could change now")
                    estimate = (smoothed_minima[-1] + smoothed_maxima[-1]) / 2
                    if self.convergence_estimate_second_order is not None and \
                            len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[
                        -1] and \
                            int(self.convergence_estimate_second_order[0]) == int(estimate):
                        self.converged = True
                        self.convergence_estimate = self.convergence_estimate_second_order[0]
                        logging.info(
                            f"converged, same 2nd order convergence estimate {estimate} as before: {self.convergence_estimate_second_order}")
                    elif self.convergence_estimate_second_order is None or self.convergence_estimate_second_order is not None \
                            and len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[
                        -1]:
                        self.convergence_estimate_second_order = [estimate, len(smoothed_minima) + len(smoothed_maxima)]
                        logging.info(
                            f"changing 2nd order convergence estimate: {self.convergence_estimate_second_order}")

    def clear_nonexistent(self):
        before_cleared = self.proposed_new_matrix.sum()
        self.proposed_new_matrix = clear_nonexistent(matrix=self.proposed_new_matrix, rhos=self.rhos)
        after_cleared = self.proposed_new_matrix.sum()
        return before_cleared - after_cleared

    @property
    def get_logging_text(self):
        return (f"time = {self.time}, population size = {self.matrix.sum()}, delta_t: {self.delta_t}, phi={self.phi}, "
                f"harshness={self.harshness}")


    def step(self, step_number: int):
        # self.delta_t = 0.001
        logging.debug(f"trying delta_t = {self.delta_t}")
        logging.debug(f"matrix at the start of the iteration:\n{self.matrix}")
        self.proposed_new_phi = update_nutrient(matrix=self.matrix,
                                                phi=self.phi,
                                                B=self.params["B"],
                                                C=self.params["C"],
                                                p=self.p,
                                                delta_t=self.delta_t)

        self.alarm_phi(self.proposed_new_phi)
        logging.debug("nutrient checked")
        death_from, dead_from_damage, dead_from_dilution = death(matrix=self.matrix,
                                                                 damage_death_rate=self.damage_death_rate,
                                                                 B=self.params["B"],
                                                                 delta_t=self.delta_t)
        grow_from, grow_to = grow(matrix=self.matrix,
                                  phi=self.phi,
                                  A=self.params["A"],
                                  r=self.params["r"], E=self.params["E"],
                                  p=self.p, delta_t=self.delta_t, q=self.q)
        accumulate_from, accumulate_to = self.accumulate_damage()
        repair_from, repair_to = repair_damage(matrix=self.matrix,
                                               r=self.params["r"],
                                               delta_t=self.delta_t,
                                               p=self.p, q=self.q)

        self.proposed_new_matrix = self.matrix - death_from - grow_from + grow_to - accumulate_from + accumulate_to \
                                   - repair_from + repair_to

        self.proposed_new_matrix = divide(matrix=self.proposed_new_matrix, q=self.q, a=self.params["a"])
        dead_from_accumulation = self.clear_nonexistent()
        self.harshness = (dead_from_damage + dead_from_accumulation)/dead_from_dilution
        logging.debug("checking combination")
        self.alarm_matrix(self.proposed_new_matrix)
        logging.debug("combination checked")
        accept_step = True
        return accept_step


class HarshnessHistory(History):
    def __init__(self, simulation_obj: HarhnessSimulation, save_path: str):
        super().__init__(simulation_obj, save_path)
        self.harshnesses = []

    def record(self) -> None:
        super().record()
        self.harshnesses.append(self.simulation.harshness)

    def get_peaks(self) -> (np.array, np.array, np.array, np.array):
        popsizes, times = np.array(self.harshnesses), np.array(self.times)
        minima, t_minima = popsizes[argrelmin(popsizes)], times[argrelmin(popsizes)]
        maxima, t_maxima = popsizes[argrelmax(popsizes)], times[argrelmax(popsizes)]
        return minima, maxima, t_minima, t_maxima

    def save(self):
        print("-------------------saving-------------------------")
        logging.info("convergence estimate " + str(self.simulation.convergence_estimate))
        if self.simulation.convergence_estimate is None:
            peaks = get_peaks(self.population_sizes)
            if convergence(peaks) in ["converged", "cycle"]:
                convergence_estimate = self.simulation.equilibrium_N(peaks)
            else:
                convergence_estimate = self.simulation.matrix.sum()
        else:
            convergence_estimate = self.simulation.convergence_estimate
            if type(convergence_estimate) == list:
                convergence_estimate = convergence_estimate[0]
        peaks = get_peaks(self.population_sizes)
        estimated_mode = convergence(peaks)
        with open(f"{self.save_path}/harshness_estimate.txt", "a") as fl:
            fl.write(f"{self.simulation.params['a']},{self.simulation.params['r']},"
                     f"{convergence_estimate},{self.simulation.converged},{estimated_mode}\n")
        with open(
                f"{self.save_path}/harshness_history_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
                "w") as fl:
            fl.write(",".join(list(map(str, self.times))) + '\n')
            fl.write(",".join(list(map(str, self.harshnesses))) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator")
    tune_parser(parser)
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)

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
                  "G": args.G, "H": args.H, "a": 0, "r": 0}
    simulation = HarhnessSimulation(params=parameters, mode=args.mode,
                            save_path=str(save_path) if args.save_path is None else args.save_path,
                            discretization_volume=args.discretization_volume,
                            discretization_damage=args.discretization_damage)
    simulation.run(1000000)
    target_harshness = simulation.convergence_estimate
    print("-------------------------")
    output_text = "error"
    if simulation.converged == False:
        output_text = "Target harshness failed to converge"
    else:
        tried_dd = [0]
        tried_harshnesses = [0]
        current_harshness = 0
        while abs(current_harshness - target_harshness) > 0.1:
            if len(tried_harshnesses) == 1:
                current_d = args.D
            else:
                reg = LinearRegression()
                reg.fit(np.array(tried_harshnesses[-2:]).reshape(2, 1),
                        np.array(tried_dd[-2:]).reshape(2, 1))
                current_d = reg.predict(np.array([[target_harshness]]))[0][0]
            test_parameters = {"A": args.A, "B": args.B, "C": args.C,
                          "D": current_d, "E": args.E, "F": 0,
                          "G": args.G, "H": args.H, "a": 0, "r": 0}
            simulation = HarhnessSimulation(params=test_parameters, mode=args.mode,
                                    save_path=str(save_path) if args.save_path is None else args.save_path,
                                    discretization_volume=args.discretization_volume,
                                    discretization_damage=args.discretization_damage)
            simulation.run(1000000)
            current_harshness = simulation.convergence_estimate
            tried_dd.append(current_d)
            tried_harshnesses.append(current_harshness)
            test_parameters["D"] = round(test_parameters["D"], 5)
            output_text = f"""Exponential Damage params: {parameters}
            f"Target harshness: {target_harshness}"
            f"Tried Ds: {tried_dd}"
            f"Tried harshnesses: {tried_harshnesses}"
            f"Best harshness: {current_harshness}"
            f"Best params: {test_parameters}"
            """
        with open(f"{save_path}/linear_da_params_with_same_harshness.txt", "w") as fl:
            fl.write(output_text)