import sys
sys.path.insert(0, '/home/blackstone/PycharmProjects/NIH/AsymmetricDivisions/simulations')
from scripts.ChemostatSimulation import Simulation, Chemostat, Cell
import numpy as np
import pytest


@pytest.mark.parametrize("volume, dilution_rate, carrying_capacity", [[v, d, c] for v in range(100, 1000, 50)
                                                                      for d in np.linspace(1, 10, 5)
                                                                      for c in range(30, 150, 10)])
def test_carrying_capacity(volume, dilution_rate, carrying_capacity):
    simulation = Simulation(Chemostat(volume, dilution_rate, 1), carrying_capacity)
    simulation.run(200)
    last150 = simulation.history.history_table.n_cells.to_numpy()[-150:]
    stable_popsize = last150[last150 > 0].mean()
    assert carrying_capacity*0.9 < stable_popsize < carrying_capacity*1.1

@pytest.mark.parametrize("critical_nutrient_amount, nutrient_accumulation_rate, chemostat_N", [[cna, nar, cN] for cna in range(1, 200, 5)
                                                                      for nar in np.linspace(1, 10, 5)
                                                                      for cN in range(1, 200)])
def test_lookup_table(critical_nutrient_amount, nutrient_accumulation_rate, chemostat_N):
    expected_age = critical_nutrient_amount / (nutrient_accumulation_rate / chemostat_N)
    if expected_age <= 2:
        assert True
    else:
        Cell.critical_nutrient_amount = critical_nutrient_amount
        Cell.nutrient_accumulation_rate = nutrient_accumulation_rate
        Cell.lambda_large_lookup = Cell.load_lookup_table_for_lambda_large()
        total_stats = []
        while len(total_stats) < 100:
            chemostat = Chemostat(volume_val=1000, dilution_rate=0, n_cells=chemostat_N)
            for cell in chemostat.cells:
                while not cell.has_reproduced:
                    cell._age += 1
                    cell.reproduce(1)
            total_stats += [cell.age for cell in chemostat.cells]
        experimental_age = np.array(total_stats).mean()
        assert expected_age * 0.9 < experimental_age < expected_age * 1.1
