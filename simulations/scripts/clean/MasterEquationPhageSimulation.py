import logging

from MasterEquationSimulation import Simulation, History, InvalidActionException
from convergence_functions import get_peaks, equilibrium_N_phages
from master_equation_phage_functions import *


class PhageSimulation(Simulation):

    def __init__(self, params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251,
                 phage_influx: float = 0,
                 ):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage)
        self.ksi = np.random.exponential(10000)
        self.history = PhageHistory(self, save_path=save_path)
        self.proposed_new_ksi = None
        self.exited_phages = 0
        self.initial_population_size = None
        self.ksi_0 = phage_influx

    @staticmethod
    def alarm_ksi(scalar: float) -> None:
        if scalar < 0:
            logging.debug(f"failed the check - {scalar} (ksi)")
            raise InvalidActionException

    def accumulate_damage(self):
        return accumulate_phage(matrix=self.matrix,
                                C=self.params["C"], D=self.params["D"], F=self.params["F"],
                                ksi=self.ksi,
                                delta_t=self.delta_t,
                                p=self.p, q=self.q)

    def clear_nonexistent(self):
        self.proposed_new_matrix, self.exited_phages = clear_nonexistent(matrix=self.proposed_new_matrix,
                                                                         rhos=self.rhos,
                                                                         death_function_threshold=self.params["T"]
                                                                         )

    def step(self, step_number: int):
        accept_step = super().step(step_number)
        self.proposed_new_ksi = update_phage(matrix=self.matrix,
                                             damage_death_rate=self.damage_death_rate,
                                             ksi=self.ksi,
                                             B=self.params["B"], C=self.params["C"], F=self.params["F"],
                                             p=self.p, q=self.q,
                                             exited_phages=self.exited_phages,
                                             ksi_0=self.ksi_0,
                                             delta_t=self.delta_t)
        self.alarm_ksi(self.proposed_new_ksi)
        return accept_step

    def upkeep_after_step(self) -> None:
        super().upkeep_after_step()
        self.ksi = self.proposed_new_ksi
        # if self.matrix.sum() < self.initial_population_size * 0.05:
        #     self.converged = True
        #     self.convergence_estimate = 0

    @property
    def get_logging_text(self):
        return (f"time = {self.time}, population size = {self.matrix.sum()}, delta_t: {self.delta_t}, phi={self.phi}, "
                f"ksi={self.ksi}")

    def prepare_to_run(self):
        self.initial_population_size = self.matrix.sum()

    def equilibrium_N(self, peaks):
        return equilibrium_N_phages(peaks)


class PhageHistory(History):
    def __init__(self, simulation_obj: PhageSimulation, save_path: str):
        super().__init__(simulation_obj, save_path)
        self.phage_history = []
        self.damage_history = []
        self.burst_dist_history = []
        self.burst_dist = None

    def record(self) -> None:
        super().record()
        self.phage_history.append(self.simulation.ksi)
        self.damage_history.append(
            (self.simulation.matrix * self.simulation.q.reshape(1, len(self.simulation.q))).sum())
        burst_dist = (self.simulation.damage_death_rate * self.simulation.matrix).sum(axis=0)
        self.burst_dist_history.append(burst_dist)
        if len(self.burst_dist_history) > 1000:
            self.burst_dist_history = self.burst_dist_history[1:]

    def prepare_to_save(self) -> None:
        super().prepare_to_save()
        if self.simulation.convergence_estimate == 0:
            ksi = 0
            dam = 0
        else:
            peaks = get_peaks(self.phage_history)
            if len(peaks) > 1:
                ksi = self.simulation.equilibrium_N(peaks)
            else:
                ksi = self.phage_history[-1]
            peaks = get_peaks(self.damage_history)
            if len(peaks) > 1:
                dam = self.simulation.equilibrium_N(peaks)
            else:
                dam = self.damage_history[-1]

        self.text += ("," + str(round(ksi, 5)) + "," + str(round(dam, 5)) + "," +
                      str(self.burst_dist_history[-1].argmax()))
        # deltas = self.times - np.array([0] + list(self.times)[:-1])
        # self.burst_dist = [history * delta for history, delta in zip(self.burst_dist_history, deltas[-1000:])]

    def save(self) -> None:
        super().save()
        # with open(f"{self.save_path}/"
        #           f"population_size_history_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
        #           "a") as fl:
        #     fl.write(",".join(list(map(str, self.phage_history))) + '\n')
        #     fl.write(",".join(list(map(str, self.damage_history ))) + '\n')
        # with open(f"{self.save_path}/"
        #           f"burst_size_dists_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
        #           "w") as fl:
        #     for el in self.burst_dist:
        #         fl.write(",".join(list(map(str, el))) + '\n')
        # with open(f"{self.save_path}/"
        #           f"pop_size_dists_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
        #           "w") as fl:
        #     fl.write(",".join(list(map(str, self.simulation.matrix.sum(axis=0)))) + '\n')
        #
        # with open(f"{self.save_path}/"
        #           f"population_structure_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
        #           "w") as fl:
        #     for i in range(self.simulation.matrix.shape[0]):
        #         fl.write(",".join(list(map(str, self.simulation.matrix[i, :]))) + '\n')
