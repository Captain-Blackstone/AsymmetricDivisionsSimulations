from MasterEquationPhageSimulation import PhageSimulation
from master_equation_pcd_functions import divide
from convergence_functions import *
import logging


class PCDSimulation(PhageSimulation):
    def __init__(self, params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage)
        self.phis = []
        self.matrices = []
    def divide(self):
        return divide(matrix=self.proposed_new_matrix, q=self.q)

    def upkeep_after_step(self):
        super().upkeep_after_step()
        self.matrix[self.rhos > 1 - self.params["a"]] = 0
        # self.phis.append(self.phi)
        # self.matrices.append(self.matix)
        # peaks = get_peaks(self.phis)
        # if len(peaks) > 5:
        #     eq_phi = (peaks[-1] + peaks[-2]) / 2
        #
        #     self.phis = []
        #     self.matrices = []
        #     self.phi = eq_phi

    def check_convergence_v2(self):
        critical_period = self.max_delta_t * 20000
        # Claiming convergence only if critical period of time passed
        if self.history.times[-1] > critical_period:
            ii = (-np.array(self.history.times) + self.history.times[-1]) < critical_period
            if len(set(np.round(np.array(self.history.population_sizes)[ii]))) == 1 and len(
                    np.round(np.array(self.history.population_sizes)[ii])) > 1:
                # Last 'critical period' of time was with the same population size
                self.converged = True
                self.convergence_estimate = self.matrix.sum()
                logging.info(f"same population size for {critical_period} time")
            else:
                minima, maxima, t_minima, t_maxima = self.history.get_peaks()
                minima, maxima, t_minima, t_maxima = minima[-min(len(minima), len(maxima)):], \
                    maxima[-min(len(minima), len(maxima)):], \
                    t_minima[-min(len(minima), len(maxima)):], \
                    t_maxima[-min(len(minima), len(maxima)):]
                if len(minima) >= 2 and len(maxima) >= 2:  # If there were more than two minima and maxima
                    estimate = (minima[-1] + maxima[-1]) / 2  # Estimate based on last two 1st order peaks
                    if self.convergence_estimate_first_order is not None and \
                            self.time > self.convergence_estimate_first_order[2] + critical_period / 4 and \
                            len(minima) + len(maxima) != self.convergence_estimate_first_order[1] and \
                            int(self.convergence_estimate_first_order[0]) == int(estimate):
                        if abs(maxima[-1] - minima[-1]) < 10:
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
                        # logging.info(
                        #     f"changing 1st order convergence estimate: {self.convergence_estimate_first_order}")
                smoothed, t_smoothed = (minima + maxima) / 2, (t_minima + t_maxima) / 2
                if len(smoothed) > 5:
                    index_array = np.where(np.round(smoothed) != np.round(smoothed)[-1])[0]
                    if len(index_array) == 0:
                        last_time = t_smoothed[0]
                    else:
                        last_time = t_smoothed[np.max(index_array) + 1]
                    if self.history.times[-1] - last_time > critical_period:
                        self.converged = True
                        self.convergence_estimate = self.matrix.sum()
                        logging.info(f"converged, same population size for {critical_period} time")
                smoothed_minima, smoothed_maxima = smoothed[argrelmin(smoothed)], smoothed[argrelmax(smoothed)]
                if len(smoothed_minima) >= 2 and len(smoothed_maxima) >= 2:
                    estimate = (smoothed_minima[-1] + smoothed_maxima[-1]) / 2
                    if (self.convergence_estimate_second_order is not None and
                            len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1]
                            and
                            int(self.convergence_estimate_second_order[0]) == int(estimate)):
                        if abs(smoothed_maxima[-1] - smoothed_minima[-1]) < 10:
                            self.converged = True
                            self.convergence_estimate = self.convergence_estimate_second_order[0]
                            logging.info(
                                f"converged, same 2nd order convergence estimate {estimate} as before: {self.convergence_estimate_second_order}")
                    elif self.convergence_estimate_second_order is None or self.convergence_estimate_second_order is not None \
                            and len(smoothed_minima) + len(smoothed_maxima) != self.convergence_estimate_second_order[-1]:
                        self.convergence_estimate_second_order = [estimate, len(smoothed_minima) + len(smoothed_maxima)]
                        logging.info(f"changing 2nd order convergence estimate: {self.convergence_estimate_second_order}")
                peaks = get_peaks(self.history.population_sizes)
                if convergence(peaks) == "cycle":
                    self.converged = True
                    self.convergence_estimate = self.equilibrium_N(peaks)
                    logging.info("got a cycle")
