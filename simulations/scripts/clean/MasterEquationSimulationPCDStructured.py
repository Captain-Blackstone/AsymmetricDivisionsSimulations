import numpy as np
import time as tm
import traceback
import logging
from tqdm import tqdm
from MasterEquationSimulation import OverTimeException, InvalidActionException
from MasterEquationPhageSimulation import PhageSimulation
from master_equation_functions import update_nutrient, grow, repair_damage, death
from master_equation_pcd_functions import divide, accumulate_phage
from master_equation_phage_functions import update_phage
from MasterEquationSimulation import gaussian_2d
from master_equation_phage_functions import clear_nonexistent
import sys  
import matplotlib.pyplot as plt


def plot_matrix_with_numbers(matrix, ax=None, title=None, filter=None, threshold=None, fmt='.2e'):
    """
    Plot a matrix as a grid with numbers inside squares.
    Optimized for large matrices (e.g., 40x1000).
    
    Args:
        matrix: 2D numpy array or list of lists
        title: Optional string for plot title
    """
    # Convert to numpy array if not already
    matrix = matrix.copy()
    matrix = np.array(matrix)
    matrix /= matrix[~np.isnan(matrix)].sum()
    if threshold:
        threshold = matrix[~np.isnan(matrix)].max()*1e-2
    if filter is not None:
        matrix[~filter] = 0    
    matrix[matrix == 0] = np.nan
    non_nan_cols = ~np.all(np.isnan(matrix), axis=0)
    if any(non_nan_cols):
        # last_valid_col = np.arange(0, len(non_nan_cols))[non_nan_cols == True].max()
        # print(last_valid_col)
        matrix = matrix[:, :min(50+1, matrix.shape[1])]
    # if threshold:
    #     last_valid_col = np.max(np.where(matrix > threshold)[1])
    #     matrix = matrix[:, :last_valid_col+1]
    rows, cols = matrix.shape
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))    
    # ax.imshow(matrix, aspect='auto', cmap='viridis')   
     
    # Get max absolute value for symmetric colormap around 0
    vmax = np.nanmax(np.abs(matrix))
    vmin = -vmax
    mappable = ax.imshow(matrix, aspect='auto', cmap='PiYG', vmax=vmax, vmin=vmin)
    plt.colorbar(mappable, ax=ax, label='Value')
    # for i in range(rows):
    #     for j in range(cols):
    #         val = matrix[i, j]
    #         if np.isnan(val):
    #             continue
    #         if threshold is None or val > threshold:
    #             text = f'{val:{fmt}}'
    #             color = 'white' if val/matrix[~np.isnan(matrix)].max() < 0.6 else 'black'
                # ax.text(j, i, "o", ha='center', va='center', 
                #         color=color, fontsize=6)
    
    if title:
        ax.set_title(title)    

    
class MultiChemostatPCDSimulation:
    def __init__(self, 
                 n_chemostats: int,
                 chemostat_params: dict,
                 save_path: str,
                 discretization_volume: int,
                 discretization_damage: int,
                 nondivision_threshold: int,
                 phage_influx: float,
                 max_time: int,
                 max_delta_t_limit: float,
                 mean_migration_waiting_time: float,
                 mean_migration_fraction: float,
                 birth_rate: float,
                 death_rate: float,
                 carrying_capacity: float,
                 ):
        """
        Initialize multiple non-interacting chemostats running in parallel
        
        Args:
            chemostat_params: List of parameter dictionaries, one for each chemostat
            Other args: Same as PCDSimulation
        """
        self.history = MultiChemostatHistory(simulation=self, save_path=save_path)
        self.chemostats = []
        self.current_time = 0

        self.chemostat_params = chemostat_params
        self.save_path = save_path
        self.discretization_volume = discretization_volume
        self.discretization_damage = discretization_damage
        self.nondivision_threshold = nondivision_threshold
        self.phage_influx = phage_influx
        
        # Create all chemostats first
        for _ in range(n_chemostats):
            chemostat = PCDSimulation(
                params=self.chemostat_params,
                save_path=self.save_path,
                mode="cluster", 
                discretization_volume=self.discretization_volume,
                discretization_damage=self.discretization_damage,
                nondivision_threshold=self.nondivision_threshold,
                phage_influx=self.phage_influx,
                multi_chemostat_simulation=self
            )
            self.chemostats.append(chemostat)
        
        # Find smallest delta_t among all chemostats for synchronization
        self.delta_t = 0.001
        self.max_delta_t = self.delta_t
        self.max_time = max_time
        self.max_delta_t_limit = max_delta_t_limit
        self.mean_migration_waiting_time = mean_migration_waiting_time
        self.mean_migration_fraction = mean_migration_fraction
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.carrying_capacity = carrying_capacity

        self.next_birth_death_time = self.get_next_birth_death_time()
        self.next_migration_time = self.get_next_migration_time()
        
            
    def step(self):
        """Run one timestep for all chemostats using the same time step"""
        logging.debug(f"trying delta_t = {self.delta_t}")
        accept_step = True
        for chemostat in self.chemostats:
            try:
                step_accepted = chemostat.step(delta_t=self.delta_t)
            except InvalidActionException:
                accept_step = False
                break
        return accept_step
        

    def run(self, n_steps: int, save=True):
        starting_time = tm.time()
        try:
            last_recorded = 0
            for step_number in tqdm(range(n_steps)):
                accept_step = False
                while not (accept_step):
                    accept_step = self.step()
                    self.check_time(starting_time=starting_time, max_time=self.max_time)
                    if not accept_step:
                        self.delta_t /= 10
                        if self.delta_t == 0:
                            logging.warning("No way to make the next step")
                            self.delta_t = 1e-20
                self.upkeep_after_step()
                if self.current_time > self.next_migration_time:
                    self.migration()
                    self.next_migration_time = self.get_next_migration_time()
                    
                if self.current_time > self.next_birth_death_time:
                    self.history.save()
                    self.birth_death()
                    self.next_birth_death_time = self.get_next_birth_death_time()
                    

                last_recorded += 1
                if last_recorded == 200:
                    self.history.record()
                    last_recorded = 0
                    logging.info(self.get_logging_text())                        
        except Exception:
            error_message = traceback.format_exc()
            logging.error(error_message)
        finally:
            self.history.record()
            if save:
                self.history.save()

    def check_time(self, starting_time: float, max_time: int):
        if tm.time() > starting_time + max_time:
            raise OverTimeException
    
    def upkeep_after_step(self):
        for chemostat in self.chemostats:
            chemostat.upkeep_after_step()
        self.current_time += self.delta_t
        self.max_delta_t = max(self.max_delta_t, self.delta_t)
        self.delta_t *= 2
        self.delta_t = min(self.delta_t, self.max_delta_t_limit)
    
    def get_logging_text(self):
        text =  f"time: {self.current_time}, delta_t: {self.delta_t} \n"
        for chemostat in self.chemostats:
            text += f"N: {round(chemostat.population_size)} ({round(100*chemostat.population_size/sum([c.population_size for c in self .chemostats]), 2)}%), %M: {round(chemostat.mutated_matrix.sum()/chemostat.population_size, 5)}, phage: {round(chemostat.ksi)}, nutrient: {round(chemostat.phi, 2)}\n"
        
        return text
    
    def migration(self):
        logging.info("MIGRATION")
        probs = np.array([chemostat.population_size for chemostat in self.chemostats])
        source_chemostat = np.random.choice(self.chemostats, p=probs/probs.sum())
        destination_chemostat = np.random.choice(self.chemostats)
        if source_chemostat is destination_chemostat:
            return
        migration_fraction = min(1, np.random.exponential(scale=self.mean_migration_fraction))
        n_migrating_bacteria = int(source_chemostat.population_size * migration_fraction)
        n_migrating_mutants = np.random.binomial(n_migrating_bacteria, source_chemostat.mutated_matrix.sum()/source_chemostat.population_size)
        n_migrating_wt = n_migrating_bacteria - n_migrating_mutants
        
        for n_migrating, matrix, dest_matrix in zip([n_migrating_wt, n_migrating_mutants], [source_chemostat.matrix, source_chemostat.mutated_matrix], [destination_chemostat.matrix, destination_chemostat.mutated_matrix]):
            if n_migrating == n_migrating_wt:
                print("wt")
            else:
                print("mutant")
            bacteria_still_needed = n_migrating
            while bacteria_still_needed > 1 and any(matrix.flatten() > 1):    
                migration_matrix = np.zeros_like(source_chemostat.matrix)
                # print("bacteria_still_needed: ", bacteria_still_needed)
                # Create a flattened copy of remaining bacteria
                remaining_matrix = matrix - migration_matrix
                # print(remaining_matrix)
                flat_matrix = remaining_matrix.flatten()
                
                # Get probabilities for each position
                probs = flat_matrix / flat_matrix.sum()
                
                chosen_indices = np.random.choice(len(flat_matrix), size=bacteria_still_needed, p=probs)
                # print("chosen_indices: ", chosen_indices)
                # Count occurrences of each position
                unique, counts = np.unique(chosen_indices, return_counts=True)
                # print("unique, counts: ", unique, counts)
                # For each chosen position

                for idx, count in zip(unique, counts):
                    # print("idx, count: ", idx, count)
                    # Convert flat index back to 2D coordinates
                    rows, cols = np.unravel_index(idx, source_chemostat.matrix.shape)
                    # Take minimum between what we want and what's available
                    available = remaining_matrix[rows, cols]
                    actual_count = min(count, int(np.floor(available)))
                    # print(rows, cols, actual_count)
                    migration_matrix[rows, cols] += actual_count
                
                # print("migration_matrix.sum(): ", migration_matrix.sum())
                # Update count of bacteria still needed
                bacteria_still_needed -= int(migration_matrix.sum())
                matrix -= migration_matrix
                dest_matrix += migration_matrix
                
    def birth_death(self):
        is_death = np.random.random() < self.death_rate/(self.birth_rate * (1 - len(self.chemostats)/self.carrying_capacity) + self.death_rate)
        if is_death:
            logging.info("DEATH")
            chemostat_to_die = np.random.choice(self.chemostats)
            self.chemostats.remove(chemostat_to_die)
            del chemostat_to_die
        else:
            logging.info("BIRTH")
            new_chemostat = PCDSimulation(
                params=self.chemostat_params,
                save_path=self.save_path,
                mode="cluster", 
                discretization_volume=self.discretization_volume,
                discretization_damage=self.discretization_damage,
                nondivision_threshold=self.nondivision_threshold,
                phage_influx=self.phage_influx,
                multi_chemostat_simulation=self)
            new_chemostat.matrix = np.zeros_like(new_chemostat.matrix)
            new_chemostat.mutated_matrix = np.zeros_like(new_chemostat.mutated_matrix)
            new_chemostat.phi = 1
            new_chemostat.ksi = self.phage_influx
            self.chemostats.append(new_chemostat)


    def get_next_birth_death_time(self):
        if self.birth_rate == 0 and self.death_rate == 0:
            return np.inf
        lambda_event = (self.birth_rate * (1 - len(self.chemostats)/self.carrying_capacity) + self.death_rate)*len(self.chemostats)
        print("waiting time, prob_death")
        print(1/lambda_event, self.death_rate/(self.birth_rate * (1 - len(self.chemostats)/self.carrying_capacity) + self.death_rate))
        next_event = self.current_time + np.random.exponential(scale=1/lambda_event)
        print("next event", next_event)
        return next_event
        


    def get_next_migration_time(self):
        next_migration = self.current_time + np.random.exponential(scale=self.mean_migration_waiting_time)
        print("next migration", next_migration)
        return next_migration
    
    

            

class MultiChemostatHistory:
    def __init__(self, simulation: MultiChemostatPCDSimulation, save_path: str):
        self.simulation = simulation
        self.save_path = Path(save_path)
        self.period = 0
        self.reset()
        
    def reset(self):
        self.phage_history = []
        self.population_history = []
        self.mutant_population_history = []
        self.time = []
        self.period += 1

    def record(self):
        """Record the current state."""
        self.time.append(self.simulation.current_time)
        # Replace loop with vectorized operation
        populations = np.array([chemostat.matrix.sum() for chemostat in self.simulation.chemostats])
        mutant_populations = np.array([chemostat.mutated_matrix.sum() for chemostat in self.simulation.chemostats])
        self.population_history.append(populations)
        self.mutant_population_history.append(mutant_populations)
        self.phage_history.append(np.array([chemostat.ksi for chemostat in self.simulation.chemostats]))

    def save(self):
        np.savetxt(self.save_path / f"{self.period}_history_wt.txt", np.array(self.population_history).T)
        np.savetxt(self.save_path / f"{self.period}_history_mutant.txt", np.array(self.mutant_population_history).T)
        np.savetxt(self.save_path / f"{self.period}_history_phage.txt", np.array(self.phage_history).T)
        np.savetxt(self.save_path / f"{self.period}_time.txt", self.time)
        logging.info("History saved to " + str(self.save_path))
        # for i, chemostat in enumerate(self.simulation.chemostats):
        #     with open(f"{self.save_path}/final_state_{i}.txt", "w") as fl:
        #         for el in chemostat.matrix:
        #             fl.write(" ".join(map(str, el)) + '\n')
            
        #     with open(f"{self.save_path}/final_state_mutant_{i}.txt", "w") as fl:
        #         for el in chemostat.mutated_matrix:
        #             fl.write(" ".join(map(str, el)) + '\n')
        with open(f"{self.save_path}/command.txt", "w") as fl:
            fl.write(" ".join(sys.argv))
        self.reset()
            



class PCDSimulation(PhageSimulation):
    def __init__(self, params: dict,
                 save_path: str,
                 mode: str,
                 multi_chemostat_simulation,
                 discretization_volume: int = 251,
                 discretization_damage: int = 251,
                 nondivision_threshold: int = 1,
                 phage_influx: float = 0,
                 ):
        super().__init__(params, save_path, mode, discretization_volume, discretization_damage, phage_influx)
        self.plotted = -1
        self.multi_chemostat_simulation = multi_chemostat_simulation
        self.nondivision_threshold = nondivision_threshold
        mean_p, mean_q, var_p, var_q, starting_popsize = \
            (1 + np.random.random(),
             np.random.random(),
             np.random.random(),
             np.random.random(),
             np.random.exponential(100000000))
        x, y = np.meshgrid(self.p, self.q)
        mut_fraction = np.random.uniform(0.5, 1)
        self.mutated_matrix = gaussian_2d(x.T, y.T, mean_p, mean_q, var_p, var_q)


        self.matrix /= self.matrix.sum()
        self.mutated_matrix /= self.mutated_matrix.sum()
        total_population = 10**5
        self.matrix *= total_population * (1-mut_fraction)
        self.mutated_matrix *= total_population * mut_fraction
        
        self.phi = 1
        self.proposed_new_mutated_matrix = None
        
        
        




    def death(self, matrix: np.ndarray):
        return death(matrix=matrix, damage_death_rate=self.damage_death_rate, B=self.params["B"], delta_t=self.delta_t)

    def accumulate_damage(self, matrix: np.ndarray):
        return accumulate_phage(matrix=matrix,
                                C=self.params["C"], F=self.params["F"], D=self.params["D"],
                                ksi=self.ksi,
                                delta_t=self.delta_t,
                                p=self.p, q=self.q)

    def divide(self, matrix: np.ndarray):
        return divide(matrix=matrix, q=self.q, nondivision_threshold=self.nondivision_threshold)

    def clear_nonexistent(self):
        self.exited_phages = 0
        self.proposed_new_matrix, exited_phages1 = clear_nonexistent(matrix=self.proposed_new_matrix,
                                                                         rhos=self.rhos,
                                                                         death_function_threshold=self.params["T"])
        self.proposed_new_mutated_matrix, exited_phages2 = clear_nonexistent(matrix=self.proposed_new_mutated_matrix,
                                                                         rhos=self.rhos,
                                                                         death_function_threshold=self.params["T"])
        self.exited_phages = exited_phages1 + exited_phages2
        

    def upkeep_after_step(self):
        super().upkeep_after_step()
        self.matrix = self.proposed_new_matrix

        self.mutated_matrix = self.proposed_new_mutated_matrix
        self.matrix[self.rhos > 1 - self.params["a"]] = 0
        self.mutated_matrix[self.rhos > 1 - self.params["a_mut"]] = 0
        

    def step(self, delta_t: float):
        total_matrix = self.matrix + self.mutated_matrix
        self.delta_t = delta_t
        self.proposed_new_phi = update_nutrient(matrix=total_matrix,
                                                phi=self.phi,
                                                B=self.params["B"],
                                                C=self.params["C"],
                                                p=self.p,
                                                delta_t=self.delta_t)

        self.alarm_phi(self.proposed_new_phi)
        logging.debug("nutrient checked")
        death_from = self.death(self.matrix)
        death_from_mutated = self.death(self.mutated_matrix)
        
        grow_from, grow_to = grow(matrix=self.matrix,
                                  phi=self.phi,
                                  A=self.params["A"],
                                  r=self.params["r"], E=self.params["E"],
                                  p=self.p, delta_t=self.delta_t, q=self.q)
        grow_from_mutated, grow_to_mutated = grow(matrix=self.mutated_matrix,
                                                  phi=self.phi,
                                                  A=self.params["A"],
                                                  r=self.params["r_mut"], E=self.params["E"],
                                                  p=self.p, delta_t=self.delta_t, q=self.q)
        
        accumulate_from, accumulate_to = self.accumulate_damage(matrix=self.matrix)
        accumulate_from_mutated, accumulate_to_mutated = self.accumulate_damage(matrix=self.mutated_matrix)
        
        repair_from, repair_to = repair_damage(matrix=self.matrix,
                                               r=self.params["r"],
                                               delta_t=self.delta_t,
                                               p=self.p, q=self.q)
        
        repair_from_mutated, repair_to_mutated = repair_damage(matrix=self.mutated_matrix,
                                               r=self.params["r_mut"],
                                               delta_t=self.delta_t,
                                               p=self.p, q=self.q)

        self.proposed_new_matrix = self.matrix - death_from - grow_from + grow_to - accumulate_from + accumulate_to \
                     - repair_from + repair_to
        self.proposed_new_mutated_matrix = self.mutated_matrix - death_from_mutated - grow_from_mutated + grow_to_mutated - accumulate_from_mutated + accumulate_to_mutated \
                     - repair_from_mutated + repair_to_mutated

        divided_new_matrix = self.divide(matrix=self.proposed_new_matrix)
        divided_new_mutated_matrix = self.divide(matrix=self.proposed_new_mutated_matrix)
        self.clear_nonexistent()
        logging.debug("checking combination")
        self.alarm_matrix(divided_new_matrix)
        self.alarm_matrix(divided_new_mutated_matrix)
        logging.debug("combination checked")
        self.proposed_new_ksi = update_phage(matrix=total_matrix,
                                             damage_death_rate=self.damage_death_rate,
                                             ksi=self.ksi,
                                             B=self.params["B"], C=self.params["C"], F=self.params["F"],
                                             p=self.p, q=self.q,
                                             exited_phages=self.exited_phages,
                                             ksi_0=self.ksi_0,
                                             delta_t=self.delta_t)
        self.alarm_ksi(self.proposed_new_ksi)
        accept_step = True


        self.proposed_new_matrix = divided_new_matrix
        self.proposed_new_mutated_matrix = divided_new_mutated_matrix
        return accept_step
    
    @property
    def population_size(self):
        return self.matrix.sum() + self.mutated_matrix.sum()


if __name__ == "__main__":
    import atexit
    import argparse
    from command_line_interface_functions import tune_parser, write_completion
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="MasterEquation simulator PCD")
    tune_parser(parser, ar_type=float)
    parser.add_argument("-a_mut", type=float, help="value of a for a mutated strain")
    parser.add_argument("-r_mut", type=float, help="value of a for a mutated strain")
    parser.add_argument("--nondivision_threshold", type=int, default=1, help="cells with number of phages >= nondivision_threshold will not divide after they grow to the division size and their growth will be arrested.")
    parser.add_argument("--phage_influx", type=float, default=0, help="psi parameter (see the paper)")
    parser.add_argument("--number_of_chemostats", type=int, default=1, help="number of chemostats in a metapopulation")
    parser.add_argument("--refine", type=float, default=0)
    parser.add_argument("-dft", "--death_function_threshold", type=float, default=1, help="T parameter (see the paper)")
    parser.add_argument("-dfc", "--death_function_curvature", type=float, default=1, help="G parameter (see the paper)")
    parser.add_argument("--save", action='store_true', help="whether the results of the simulation have to be saved")
    parser.add_argument("--max_time", type=int, help="maximum time of the simulation in seconds")
    parser.add_argument("--max_delta_t_limit", type=float, default=0.001, help="maximum delta_t of the simulation")
    parser.add_argument("--mean_migration_waiting_time", type=float, default=np.inf, help="lambda of exponential distribution of migration time")
    parser.add_argument("--mean_migration_fraction", type=float, default=0, help="mean fraction of the population that migrates")
    parser.add_argument("--birth_rate", type=float, default=0, help="birth rate")
    parser.add_argument("--death_rate", type=float, default=0, help="death rate")
    parser.add_argument("--carrying_capacity", type=int, default=10, help="carrying capacity")

    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)
    if not args.save_path:
        save_path = f"../../data/master_equation/" \
                f"{args.A}_{args.B}_{args.C}_{args.D}_{args.phage_influx}_{args.E}_{args.F}_{args.death_function_threshold}_{args.death_function_curvature}"
    else:
        save_path = args.save_path
    Path(save_path).mkdir(exist_ok=True)
    atexit.register(lambda: write_completion(save_path))
    simulation = MultiChemostatPCDSimulation(n_chemostats=args.number_of_chemostats,
                                             chemostat_params={"A": args.A, "B": args.B, "C": args.C, "D": args.D,
                                                               "E": args.E, "F": args.F,
                                                               "G": args.death_function_curvature, "T": args.death_function_threshold,
                                                               "a": args.a, "r": args.r, "a_mut": args.a_mut, "r_mut": args.r_mut},
                                                     save_path=save_path,
                                                     discretization_volume=args.discretization_volume,
                                                     discretization_damage=args.discretization_damage,
                                                     nondivision_threshold=args.nondivision_threshold,
                                                     phage_influx=args.phage_influx,
                                                     max_time=args.max_time,
                                                     max_delta_t_limit=args.max_delta_t_limit,
                                                     mean_migration_waiting_time=args.mean_migration_waiting_time,
                                                     mean_migration_fraction=args.mean_migration_fraction,
                                                     birth_rate=args.birth_rate,
                                                     death_rate=args.death_rate,
                                                     carrying_capacity=args.carrying_capacity
                                                     )
    simulation.run(n_steps=args.niterations, save=args.save)


