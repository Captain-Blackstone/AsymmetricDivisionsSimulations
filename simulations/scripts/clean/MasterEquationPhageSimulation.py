import logging

from MasterEquationSimulation import Simulation, History, InvalidActionException
from convergence_functions import get_peaks, equilibrium_N_phages
from master_equation_phage_functions import *


class PhageSimulation(Simulation):

    def __init__(self, params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage)
        self.ksi = np.random.exponential(10000)
        self.history = PhageHistory(self, save_path=save_path)
        self.proposed_new_ksi = None
        self.exited_phages = 0
        self.initial_population_size = None

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
                                                                         rhos=self.rhos)

    def step(self, step_number: int):
        accept_step = super().step(step_number)
        self.proposed_new_ksi = update_phage(matrix=self.matrix,
                                             damage_death_rate=self.damage_death_rate,
                                             ksi=self.ksi,
                                             B=self.params["B"], C=self.params["C"], F=self.params["F"],
                                             p=self.p, q=self.q,
                                             exited_phages=self.exited_phages,
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

    def run(self, n_steps: int, save=True) -> (np.array, float):
        super().run(n_steps, save)
        return self.matrix, self.phi, self.ksi


class PhageHistory(History):
    def __init__(self, simulation_obj: PhageSimulation, save_path: str):
        super().__init__(simulation_obj, save_path)
        self.phage_history = []
        self.damage_history = []

    def record(self) -> None:
        super().record()
        self.phage_history.append(self.simulation.ksi)
        self.damage_history.append((self.simulation.matrix * self.simulation.q.reshape(1, len(self.simulation.q))).sum())

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

        self.text += "," + str(round(ksi, 5)) + "," + str(round(dam, 5))

    def save(self) -> None:
        super().save()
        # with open(f"{self.save_path}/"
        #           f"population_size_history_{self.simulation.params['a']}_{self.simulation.params['r']}.txt",
        #           "a") as fl:
        #     fl.write(",".join(list(map(str, self.phage_history))) + '\n')
        #     fl.write(",".join(list(map(str, self.damage_history ))) + '\n')
