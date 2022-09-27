import sys
sys.path.insert(0, '/home/blackstone/PycharmProjects/NIH/AsymmetricDivisions/simulations')
from scripts.ChemostatSimulation import Simulation, Chemostat
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
