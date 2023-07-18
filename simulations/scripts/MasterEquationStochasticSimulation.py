from itertools import filterfalse
import numpy as np
import warnings


class Chemostat:
    def __init__(self,
                 dilution_rate: float,
                 n_cells=None,
                 matrix_shape: np.array = (41, 41),
                 starting_nutrient_concentration: float = 1,
                 asymmetry: float = 0.0,
                 repair: float = 0.0):
        self.B = dilution_rate
        self.nutrient_concentration = starting_nutrient_concentration
        self._cells = []
        self._n = 0
        if n_cells:
            self.populate_with_cells(n_cells, asymmetry=asymmetry, repair=repair)
        self.matrix = np.zeros(matrix_shape)
        self.p = np.linspace(1, 2, matrix_shape[0])
        self.q = np.linspace(0, 1, matrix_shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rhos = np.outer(1 / self.p, self.q)
            self.damage_death_rate = (self.rhos / (1 - self.rhos))
        self.damage_death_rate[np.isinf(self.damage_death_rate)] = self.damage_death_rate[
            ~np.isinf(self.damage_death_rate)].max()
        self._n_array = None

    @property
    def N(self) -> int:
        return self._n

    @property
    def cells(self) -> list:
        return self._cells

    def populate_with_cells(self, n_cells: int, asymmetry: float, repair: float) -> None:
        self._cells += [Cell(chemostat=self,
                             asymmetry=asymmetry,
                             repair=repair) for _ in range(n_cells)]
        self._n = len(self._cells)

    def cells_from_n_array(self):
        if self.n_array is not None:
            self.cells = []
            for i in range(self.matrix.shape[0]):
                for j in range(self.matrix.shape[1]):
                    n_cells = round(self.matrix[i, j])
                    self.cells += [Cell(chemostat=self,
                                        nutrient=self.matrix.shape[0] - 1 + i,
                                        damage=j,
                                        asymmetry=0,
                                        repair=0) for _ in range(n_cells)]
            self.n_array = None

    def dilute(self, time_step_duration: float) -> None:
        expected_n_cells_to_remove = self.B * self.N
        n_cells_to_remove = np.random.poisson(expected_n_cells_to_remove * time_step_duration, 1)[0]
        choose_from = list(filterfalse(lambda c: c.has_died, self.cells))
        dead_cells = np.random.choice(choose_from,
                                      size=min(len(choose_from), n_cells_to_remove), replace=False)
        for cell in dead_cells:
            cell.die(cause="dilution")
        self._n = len(list(filterfalse(lambda c: c.has_died, self.cells)))
        # self.nutrient_concentration += self.B * (1 - self.nutrient_concentration) * time_step_duration

    @cells.setter
    def cells(self, cells):
        self._cells = cells
        self._n = len(self._cells)

    def set_n_array(self):
        if self._n_array is None:
            self._n_array = np.zeros_like(self.matrix)
            damage_step = 1 / (len(self.matrix) - 1)
            np.add.at(self._n_array,
                      np.digitize([cell.damage_concentration for cell in self.cells],
                                  self.matrix + damage_step / 2),
                      1)

    @property
    def n_array(self):
        return self._n_array

    @n_array.setter
    def n_array(self, n_array: np.array):
        self._n_array = n_array
        if n_array is not None:
            self._n = n_array.sum()


class Cell:
    # Nutrient accumulation
    critical_nutrient_amount = 80
    nutrient_to_volume_scaling_factor = 1 / 40

    maximum_damage_amount = 40

    # mutation rates
    asymmetry_mutation_rate = 0
    asymmetry_mutation_step = 0.01
    repair_mutation_rate = 0
    repair_mutation_step = 0.01

    def __init__(self,
                 chemostat: Chemostat,
                 nutrient=round(critical_nutrient_amount / 2),
                 growth_rate=28.0,
                 damage_accumulation_linear_component=0.0,
                 repair_cost_coefficient=1.0,
                 damage_accumulation_exponential_component=0.0,
                 damage_survival_dependency=1.0,
                 damage=0.0,
                 asymmetry=0.0,
                 repair=0.0):
        self.chemostat = chemostat


        self.nutrient = nutrient
        self.damage = damage

        self.growth_rate = growth_rate  # A
        self.damage_accumulation_linear_component = damage_accumulation_linear_component  # D
        self.repair_cost_coefficient = repair_cost_coefficient  # E
        self.damage_accumulation_exponential_component = damage_accumulation_exponential_component  # F
        self.damage_survival_dependency = damage_survival_dependency  # G

        self.asymmetry = asymmetry
        self.repair = repair

        self.has_reproduced = False
        self._has_died = ""

        self.recently_accumulated_damage = 0
        self.recently_accumulated_nutrient = 0

    def choose_damage_to_accumulate(self, time_step_duration):
        # print("F", ((1+self.damage_accumulation_exponential_component)**time_step_duration - 1))
        expected_damage_to_accumulate = ((1+self.damage_accumulation_exponential_component)**time_step_duration - 1) * self.damage + \
                                        self.damage_accumulation_linear_component * (self.maximum_damage_amount+1) * self.volume
        prob_accumulation = expected_damage_to_accumulate * time_step_duration
        if self.repair > 0:
            expected_damage_to_repair = self.repair * (self.maximum_damage_amount+1) * self.volume
            prob_repair = expected_damage_to_repair * time_step_duration
        else:
            expected_damage_to_repair = 0
            prob_repair = 0
        self.recently_accumulated_damage = int(np.random.uniform(0, 1) < prob_accumulation) - \
                                           int(np.random.uniform(0, 1) < prob_repair)

        # self.recently_accumulated_damage = np.random.poisson(expected_damage_to_accumulate*time_step_duration)[0] - \
        #                                    np.random.poisson((expected_damage_to_repair*time_step_duration))[0]

    def choose_nutrient_to_accumulate(self, time_step_duration: float) -> None:
        expected_nutrient = self.growth_rate * (1 - self.repair / self.repair_cost_coefficient) * \
                            self.volume * \
                            self.chemostat.nutrient_concentration
        prob = expected_nutrient * time_step_duration  # poisson(0)
        self.recently_accumulated_nutrient = int(np.random.uniform(0, 1) < prob)
        # self.recently_accumulated_nutrient = np.random.poisson(expected_nutrient*time_step_duration, 1)[0]

    def live(self, time_step_duration: float) -> None:
        self.nutrient += self.recently_accumulated_nutrient
        self.damage += self.recently_accumulated_damage

        # -rho/(1-rho) * n * delta_t
        working_concentration = self.damage_concentration if self.damage_concentration < 1 else \
            (self.maximum_damage_amount-1)/self.maximum_damage_amount / \
            ((self.critical_nutrient_amount/2)*self.nutrient_to_volume_scaling_factor)
        if np.random.uniform(0, 1) < time_step_duration * (
                working_concentration / (1 - working_concentration)) ** self.damage_survival_dependency:
            self.die(cause="damage")

    def reproduce(self) -> list:
        self.has_reproduced = self.volume >= self.critical_volume
        if self.has_reproduced:
            offspring_asymmetry = self.asymmetry
            if np.random.uniform() < self.asymmetry_mutation_rate:
                offspring_asymmetry += np.random.choice([self.asymmetry_mutation_step, -self.asymmetry_mutation_step])
            offspring_repair = self.repair
            if np.random.uniform() < self.repair_mutation_rate:
                offspring_repair += np.random.choice([self.repair_mutation_step, -self.repair_mutation_step])
            damage1 = round(self.damage * (1 + self.asymmetry) / 2)
            damage2 = self.damage - damage1
            res = [Cell(chemostat=self.chemostat,
                        growth_rate=self.growth_rate,
                        damage_accumulation_linear_component=self.damage_accumulation_linear_component,
                        repair_cost_coefficient=self.repair_cost_coefficient,
                        damage_accumulation_exponential_component=self.damage_accumulation_exponential_component,
                        damage_survival_dependency=self.damage_survival_dependency,
                        asymmetry=offspring_asymmetry,
                        repair=offspring_repair,
                        damage=damage1),
                   Cell(chemostat=self.chemostat,
                        growth_rate=self.growth_rate,
                        damage_accumulation_linear_component=self.damage_accumulation_linear_component,
                        repair_cost_coefficient=self.repair_cost_coefficient,
                        damage_accumulation_exponential_component=self.damage_accumulation_exponential_component,
                        damage_survival_dependency=self.damage_survival_dependency,
                        asymmetry=offspring_asymmetry,
                        repair=offspring_repair,
                        damage=damage2),
                   ]
        else:
            res = [self]
        return res

    @property
    def damage_concentration(self) -> float:
        return (self.damage / self.maximum_damage_amount) / self.volume

    @property
    def volume(self):
        return self.nutrient * self.nutrient_to_volume_scaling_factor

    @property
    def critical_volume(self):
        return self.critical_nutrient_amount * self.nutrient_to_volume_scaling_factor

    def die(self, cause: str) -> None:
        self._has_died = cause

    @property
    def asymmetry(self) -> float:
        """
        :return: the asymmetry value of a cell. Impacts the damage inheritance of the daughter cells.
        The inherited damage for daughter cells is calculated as damage*(1+asymmetry)/2 and damage*(1-asymmetry)/2
        """
        return self._asymmetry

    @property
    def damage(self) -> float:
        """
        :return: amount of somatic damage accumulated by the cell
        """
        return self._damage

    @property
    def nutrient(self) -> int:
        return self._nutrient

    @property
    def repair(self) -> float:
        return self._repair

    @property
    def has_reproduced(self) -> bool:
        """
        :return: if cell has reproduced at the current time step
        """
        return self._has_reproduced

    @property
    def has_died(self) -> str:
        """
        :return: if cell has died at the current time step
        """
        return self._has_died

    @damage.setter
    def damage(self, damage):
        self._damage = min(self.maximum_damage_amount, max(damage, 0))

    @nutrient.setter
    def nutrient(self, nutrient):
        self._nutrient = nutrient

    @nutrient.getter
    def nutrient(self):
        return self._nutrient

    @has_reproduced.setter
    def has_reproduced(self, value: bool):
        self._has_reproduced = value

    @asymmetry.setter
    def asymmetry(self, asymmetry):
        self._asymmetry = min(max(asymmetry, 0), 1)

    @repair.setter
    def repair(self, repair):
        self._repair = min(1, max(repair, 0))

    def __repr__(self):
        return f"Damage: {self.damage}, " \
               f"Alive: {not self.has_died}, Reproduced: {self.has_reproduced}"


class Simulation:
    def __init__(self, params, mode="local"):
        self.mode = mode
        self.params = params
        self.chemostat = Chemostat(dilution_rate=self.params["B"], starting_nutrient_concentration=1)#np.random.uniform(0, 1))
        for _ in range(1000):
            self.chemostat.cells.append(Cell(chemostat=self.chemostat,
                                             growth_rate=self.params["A"],
                                             damage_accumulation_linear_component=self.params["D"],
                                             repair_cost_coefficient=self.params["E"],
                                             damage_accumulation_exponential_component=self.params["F"],
                                             damage_survival_dependency=self.params["G"],
                                             # nutrient=np.random.randint(Cell.critical_nutrient_amount/2,
                                                                        # Cell.critical_nutrient_amount),
                                             damage=0, #np.random.randint(0, Cell.maximum_damage_amount),
                                             asymmetry=self.params["a"],
                                             repair=self.params["r"]))
        self.chemostat._n = len(self.chemostat._cells)

    def step(self, time_step_duration: float, delta_time_step: float) -> float:
        accept_step = False
        increase_time_step = False #np.random.uniform(0, 1) < 0.01
        while not accept_step:
            accept_step = True
            for cell in self.chemostat.cells:
                cell.choose_nutrient_to_accumulate(time_step_duration)
                cell.choose_damage_to_accumulate(time_step_duration)
            suggested_damage_concentrations = [max(cell.damage + cell.recently_accumulated_damage, 0) /
                                               ((cell.nutrient + cell.recently_accumulated_nutrient) *
                                                cell.nutrient_to_volume_scaling_factor)
                                               for cell in self.chemostat.cells]

            if any([dc > 1 for dc in suggested_damage_concentrations]):
                print(1)
            if any([dc != 1 and dc / (1 - dc) * time_step_duration > 1 for dc in suggested_damage_concentrations]):
                print(2, [dc / (1 - dc) * time_step_duration for dc in suggested_damage_concentrations])
            if self.params["C"] * self.chemostat.nutrient_concentration * \
                    sum([cell.recently_accumulated_nutrient for cell in self.chemostat.cells]) > \
                    self.chemostat.nutrient_concentration:
                print(3, self.chemostat.N, time_step_duration)
            if any([dc != 1 and dc / (1 - dc) * time_step_duration > 1 for dc in suggested_damage_concentrations]) or \
                    self.params["C"] * self.chemostat.nutrient_concentration * \
                    sum([cell.recently_accumulated_nutrient for cell in self.chemostat.cells]) > \
                    self.chemostat.nutrient_concentration:
                accept_step = False
                increase_time_step = False
                time_step_duration -= time_step_duration * delta_time_step

        # Time passes
        for cell in self.chemostat.cells:
            cell.live(time_step_duration)

        self.chemostat.nutrient_concentration += (self.params["B"] * (1 - self.chemostat.nutrient_concentration) - \
                                                 self.params["C"] * self.chemostat.nutrient_concentration * \
                                                 sum([cell.volume for cell in self.chemostat.cells])) * time_step_duration


        # Cells are diluted
        self.chemostat.dilute(time_step_duration)

        # Alive cells reproduce
        new_cells = []
        for cell_obj in self.chemostat.cells:
            offspring = cell_obj.reproduce()
            offspring[0]._has_died = cell_obj._has_died
            new_cells += offspring
        new_cells = list(filterfalse(lambda cell_obj: cell_obj.has_died, new_cells))

        # Move to the next time step
        self.chemostat.cells = new_cells
        time_step_duration += time_step_duration * delta_time_step * int(increase_time_step)
        return time_step_duration


# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from MasterEquationSimulation import Simulation as MSimulation
# from interactive_mode import non_blocking_pause
# import atexit
#
# params = {"A": 28, "B": 0.55, "C": 1e-5, "D": 0.1, "E": 1, "F": 1, "G": 1, "a": 0, "r": 0.05}
# # params = {"A": 0, "B": 0, "C": 1e-5, "D": 0, "E": 1, "F": 10, "G": 1, "a": 0, "r": 0}
#
# simulation = Simulation(params)
# msimulation = MSimulation(params=params, mode="local",
#                                     save_path="deleteme/",
#                                     discretization_volume=41,
#                                     discretization_damage=41)
# msimulation.matrix = np.zeros_like(msimulation.matrix)
# for i in range(len(simulation.chemostat.cells)):
#     damage = np.random.randint(0, 40)
#     simulation.chemostat.cells[i].damage = damage
#     msimulation.matrix[0, damage] += 1
# msimulation.phi = 1
# n_d = []
# n_c = []
# phis_d = []
# phis_c = []
# t_d, t_c = [], []
# fig, ax = plt.subplots(2, 2)
# fig.canvas.manager.set_window_title('B=0.55, D=0.1, F=1, r=0.05')
# layer_d, = ax[1, 0].plot([0], [0], color="blue")
# layer_c, = ax[1, 0].plot([0], [0], color="red")
# layer_p_d, = ax[1, 1].plot([0], [0], color="blue")
# layer_p_c, = ax[1, 1].plot([0], [0], color="red")
# plt.show(block=False)
# atexit.register(plt.close)
#
# for i in tqdm(range(500000)):
#     simulation.step(0.01, 0)
#     msimulation.step(i)
#     n_d.append(simulation.chemostat.N)
#     n_c.append(msimulation.matrix.sum())
#     phis_d.append(simulation.chemostat.nutrient_concentration)
#     phis_c.append(msimulation.phi)
#     if i % 10 == 0:
#         print(simulation.chemostat.N, msimulation.matrix.sum())
#     # print(np.mean([cell.volume for cell in simulation.chemostat.cells]))
#     # print(sum([cell.recently_accumulated_nutrient for cell in simulation.chemostat.cells])/simulation.chemostat.N)
#     cells = np.zeros((41, 41))
#     for cell in simulation.chemostat.cells:
#         if cell.damage <= 40:
#             cells[cell.nutrient-40, cell.damage] += 1
#     if i % 1 == 0:
#         print('-----')
#
#         print(simulation.chemostat.nutrient_concentration, msimulation.phi)
#         ax[0, 0].clear()
#         ax[0, 0].imshow(cells)
#         ax[0, 1].clear()
#         ax[0, 1].imshow(msimulation.matrix)
#         layer_d.set_ydata(n_d)
#         layer_d.set_xdata(np.array(list(range(len(n_d))))*0.01)
#         layer_c.set_ydata(n_c)
#         layer_c.set_xdata(np.array(list(range(len(n_d))))*0.01)
#         layer_p_d.set_ydata(phis_d)
#         layer_p_d.set_xdata(np.array(list(range(len(n_d))))*0.01)
#         layer_p_c.set_ydata(phis_c)
#         layer_p_c.set_xdata(np.array(list(range(len(n_d))))*0.01)
#         ax[1, 0].relim()
#         ax[1, 0].autoscale_view(tight=True)
#         ax[1, 1].relim()
#         ax[1, 1].autoscale_view(tight=True)
#         non_blocking_pause(0.01)
#
#         # ax[1, 0].plot(n_d)
#         # ax[1, 0].plot(n_c)
#         # ax[1, 1].plot(phis_d)
#         # ax[1, 1].plot(phis_c)
#
#         # plt.show()
#         death_probs_d = np.zeros(len(cells))
#         for cell in simulation.chemostat.cells:
#             if cell.damage_concentration < 1 and cell.volume > 1:
#                 death_prob = 0.01 * cell.damage_concentration/(1-cell.damage_concentration)
#                 death_probs_d[cell.damage] = death_prob
#         damages = msimulation.matrix[1, :].reshape(1, 41).round(2)[0]
#         death_probs_c = msimulation.damage_death_rate[1, :] * 0.01
#         # plt.plot(death_probs_d)
#         # plt.plot(death_probs_c)
#         # input()

