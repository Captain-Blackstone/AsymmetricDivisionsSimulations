from MasterEquationPhageSimulation import PhageSimulation
from master_equation_pcd_functions import divide


class PCDSimulation(PhageSimulation):
    def __init__(self, params: dict,
                 save_path: str,
                 mode: str,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251,
                 nondivision_threshold: int = 1,
                 ):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage)
        self.nondivision_threshold = nondivision_threshold

    def divide(self):
        return divide(matrix=self.proposed_new_matrix, q=self.q, nondivision_threshold=self.nondivision_threshold)

    def upkeep_after_step(self):
        super().upkeep_after_step()
        self.matrix[self.rhos > 1 - self.params["a"]] = 0

