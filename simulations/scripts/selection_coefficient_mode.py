import pandas as pd

from ChemostatSimulation import SimulationThread, Chemostat, Cell
import json
import logging
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing
from tqdm import tqdm
import os
import datetime


class FixationSimulation:
    def __init__(self,
                 parameters: dict,
                 save_path: str,
                 mode: str,
                 n_threads: int, setup: dict):
        self.save_path = save_path
        self.mode = mode
        self.set_cell_params(parameters)
        self.threads = [FixationSimulatonThread(
                                        chemostat_obj=Chemostat(
                                             volume_val=parameters["chemostat_parameters"]["volume"],
                                             dilution_rate=parameters["chemostat_parameters"]["dilution_rate"],
                                             n_cells=100,
                                             asymmetry=setup["wt"]["asymmetry"],
                                             damage_repair_intensity=setup["wt"]["repair"]
                                         ),
                                        thread_id=i,
                                        save_path=save_path, setup=setup) for i in range(n_threads)]
        self.df = pd.DataFrame(
            columns=["wt_a", "wt_r", "mut_a", "mut_r", "population_start", "population_end", "winner", "n_steps"])

    def run_thread(self, thread_number: int) -> None:
        self.threads[thread_number].run()

    def run(self) -> None:
        for thread in self.threads:
            thread.run()
            self.df = pd.concat([self.df, pd.DataFrame.from_dict(thread.result)], ignore_index=True)
            self.df.to_csv(self.save_path, sep="\t", index=False)

    def set_cell_params(self, parameters):
        Cell.damage_repair_mode = parameters["cell_parameters"]["damage_repair_mode"]
        Cell.damage_accumulation_exponential_component = \
            parameters["cell_parameters"]["damage_accumulation_exponential_component"]
        Cell.damage_accumulation_linear_component = \
            parameters["cell_parameters"]["damage_accumulation_linear_component"]
        Cell.damage_survival_dependency = parameters["cell_parameters"]["damage_survival_dependency"]
        Cell.damage_reproduction_dependency = parameters["cell_parameters"]["damage_reproduction_dependency"]

        Cell.nutrient_accumulation_rate = parameters["cell_parameters"]["nutrient_accumulation_rate"]
        Cell.critical_nutrient_amount = parameters["cell_parameters"]["critical_nutrient_amount"]
        Cell.asymmetry_mutation_step = parameters["cell_parameters"]["asymmetry_mutation_step"]
        Cell.asymmetry_mutation_rate = parameters["cell_parameters"]["asymmetry_mutation_rate"]
        Cell.repair_mutation_step = parameters["cell_parameters"]["repair_mutation_step"]
        Cell.repair_mutation_rate = parameters["cell_parameters"]["repair_mutation_rate"]
        Cell.repair_cost_coefficient = parameters["cell_parameters"]["repair_cost_coefficient"]
        if self.mode == "cluster":
            Cell.lambda_large_lookup_path = "./lambda_large_lookup_table.csv"
        Cell.lambda_large_lookup = Cell.load_lookup_table_for_lambda_large()


class FixationSimulatonThread(SimulationThread):
    def __init__(self, chemostat_obj, thread_id, save_path, setup: dict):
        """

        :param chemostat_obj:
        :param save_path:
        :param setup: dictionary looking as following:
        {
            wt: {"asymmetry": float,
                 "repair": float},
            mut: {"asymmetry": float,
                  "repair": float},
            N: int
        }
        """
        self.save_path = save_path
        self.setup = setup
        super().__init__(run_id="",
                         run_name="",
                         thread_id=0,
                         chemostat_obj=chemostat_obj,
                         changing_environment_prob=0,
                         harsh_environment_frac=1,
                         mode="local",
                         save_path="",
                         write_cells_table=False,
                         record_history=False)
        self.run_length = 0
        self.thread_id = thread_id
        self.popsizes = []
        self.skipfirst = 5000
        self.incept = None
        self.starting_popsize = None
        self.result = None

    def run(self, n_steps=0):
        np.random.seed((os.getpid() * int(datetime.datetime.now().timestamp()) % 123456789))

        while True:
            if self.chemostat.N == 0:
                break
            if self.incept is False and \
                    len(set([cell.asymmetry for cell in self.chemostat.cells])) == 1 and  \
                    len(set([cell.damage_repair_intensity for cell in self.chemostat.cells])) == 1:
                break
            self.step(self.run_length)
            if self.incept is None:
                self.popsizes.append(self.chemostat.N)
                if len(self.popsizes) > self.skipfirst + 10:
                    x = np.arange(len(self.popsizes) - self.skipfirst).reshape((-1, 1))
                    y = np.array(self.popsizes[self.skipfirst:])
                    coef = LinearRegression().fit(x, y).coef_
                    if abs(coef) < 1e-4:
                        self.incept = True
            if self.incept:
                for cell in self.chemostat.cells:
                    if cell.age == 1:
                        cell.asymmetry = self.setup['mut']['asymmetry']
                        cell.damage_repair_intensity = self.setup['mut']['repair']
                        self.incept = False
                        self.starting_popsize = self.chemostat.N
                        break
            if self.incept == False:
                self.run_length += 1
        self.record_results()

    def record_results(self):
        if self.chemostat.N == 0:
            winner = 'none'
        elif self.chemostat.cells[0].asymmetry == self.setup['wt']['asymmetry'] \
                and self.chemostat.cells[0].damage_repair_intensity == self.setup['wt']['repair']:
            winner = 'wt'
        elif self.chemostat.cells[0].asymmetry == self.setup['mut']['asymmetry'] \
                and self.chemostat.cells[0].damage_repair_intensity == self.setup['mut']['repair']:
            winner = 'mut'
        else:
            winner = f"something went wrong, " \
                     f"a = {self.chemostat.cells[0].asymmetry}, r = {self.chemostat.cells[0].damage_repair_intensity}"
        self.result = {"wt_a": [self.setup["wt"]["asymmetry"]], "wt_r": [self.setup["wt"]["repair"]],
               "mut_a": [self.setup["mut"]["asymmetry"]], "mut_r": [self.setup["mut"]["repair"]],
               "n_steps": [self.run_length], "winner": [winner], "population_start": [self.starting_popsize],
               "population_end": [self.chemostat.N]}




parser = argparse.ArgumentParser(prog="SC")
parser.add_argument("--params", type=str, required=True)
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--mode", type=str, choices=["local", "cluster"], default="local")

parser.add_argument("--wta", type=float, required=True)
parser.add_argument("--wtr", type=float, required=True)
parser.add_argument("--muta", type=float, required=True)
parser.add_argument("--mutr", type=float, required=True)
parser.add_argument("-nt", "--nthreads", type=int, default=1)


args = parser.parse_args()
with open(args.params, "r") as fl:
    parameters = json.load(fl)

logging.info(f"Reading arguments from {args.params}.")
if parameters["changing_environment_probability"] != 0:
    logging.info("Setting changing_environment_probability to 0")
    parameters["changing_environment_probability"] = 0
if parameters["cell_parameters"]["asymmetry_mutation_rate"] != 0:
    logging.info("Setting asymmetry_mutation_rate to 0")
    parameters["cell_parameters"]["asymmetry_mutation_rate"] = 0
if parameters["cell_parameters"]["repair_mutation_rate"] != 0:
    logging.info("Setting repair_mutation_rate to 0")
    parameters["cell_parameters"]["repair_mutation_rate"] = 0

setup = {
    "wt": {"asymmetry": args.wta,
           "repair": args.wtr},
    "mut": {"asymmetry": args.muta,
            "repair": args.mutr}
}
if args.mode == "local":
    root_path = "../data/selection_coefficients"
elif args.mode == "cluster":
    root_path = "./selection_coefficients"
else:
    root_path = "./"

Path(root_path).mkdir(exist_ok=True)
save_path = f"{root_path}/{args.folder}/"
Path(save_path).mkdir(exist_ok=True)
# Write parameters needed to identify simulation
with open(f"{save_path}/params.json", "w") as fl:
    json.dump(parameters, fl)

save_path += f"a_{setup['wt']['asymmetry']}_r_{setup['wt']['repair']}_vs_a_{setup['mut']['asymmetry']}_r_{setup['mut']['repair']}.tsv"

simulation = FixationSimulation(parameters=parameters,
                                save_path=save_path,
                                n_threads=args.nthreads,
                                mode="local", setup=setup)
simulation.run()

