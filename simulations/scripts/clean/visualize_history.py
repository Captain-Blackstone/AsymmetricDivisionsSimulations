import numpy as np
import matplotlib.pyplot as plt


path = "../../data/master_equation/28.0_0.1_1e-05_0.0_1000000.0_0.1_1000.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.1_1e-05_0.0_1000000.0_0.1_0.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.1_1e-06_0.0_1000000.0_0.1_0.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_1200.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_2200.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_3200.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_4200.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_5200.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-07_0.0_100000000.0_0.0375_4000.0_1.0_1.0"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_1200.0_1.0_1.0"
path = "start_60_40"
path = "start_60_40_fixed_psi_1e9"
path = "start_60_40_fixed_psi_1e10"
path = "start_1_99_fixed_psi_1e10"
path = "start_1_99_fixed_psi_1e10_mutant_adv_a_1.1"
path = "start_1_99_fixed_psi_1e10_a_0.9999"
path = "start_1_99_fixed_psi_1e10_a_0.999"
path = "start_1_99_fixed_psi_1e10_a_0.99"
# path = "start_1_99_fixed_psi_1e10_a_0.95"
# path = "start_1_99_fixed_psi_1e10_a_0.97"
# path = "start_1_99_a_0.99"
# path = "start_1_99_a_0.99_wt_mut_have_same_wt_matrix"
# path = "start_1_99_a_0.99_wt_mut_have_same_mut_matrix"
# path = "start_1_99_a_0.99_wt_mut_have_same_mut_matrix_fixed_psi_1e10"
path = "start_1_99_fixed_psi_1e10_a_0.99" # A
path = "start_1_99_a_0.95_psi_start_1e10" # B
path = "start_1_99_a_0.99_psi_start_1e10" # C
path = "start_1_99_fixed_psi_1e10_a_0.95" # D
path = "no_mutant_a_0.99" # E
path = "no_mutant_a_0.99_fixed_psi_1e10" # F
path = "no_wt" # G
path = "no_wt_fixed_psi_1e10" # H
path = "start_1_99_a_0.99_psi_start_1e10_part_3" # C2
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_1200.0_1.0_1.0"
path = "single_chemostat_v1"
path = "../../data/master_equation/28.0_0.05_1e-08_0.0_100000000.0_0.0125_1200.0_1.0_1.0"
# path = "simpson_paradox_v1" # failed
# path = "simpson_paradox_v2" # failed
# path = "simpson_paradox_v3 # failed"
path = "simpson_paradox_v4"
path = "simpson_paradox_v5"
time_path=path + '/time.txt'
# visualize_from_files(history_wt_path=path + '/history_wt.txt',
#                      history_mutant_path=path + '/history_mutant.txt',
#                      time_path=time_path)
# visualize_from_files_per_chemostat(history_wt_path=path + '/history_wt.txt',
#                      history_mutant_path=path + '/history_mutant.txt',
#                      time_path=time_path)

import os

def visualize_all_populations(path, save=False, save_path=None, show=True):
    # Get list of files in directory
    files = os.listdir(path)
    files = sorted(files)
    
    # Initialize lists to store data
    wt_data = []
    mutant_data = []
    phage_data = []
    time_data = []
    
    # Loop over files and load data
    for file in files:
        if file.endswith('_time.txt'):
            time_data.append(np.loadtxt(os.path.join(path, file)))
        elif file.endswith('_history_wt.txt'):
            wt_data.append(np.loadtxt(os.path.join(path, file)))
        elif file.endswith('_history_mutant.txt'):
            mutant_data.append(np.loadtxt(os.path.join(path, file)))
        elif file.endswith('_history_phage.txt'):
            phage_data.append(np.loadtxt(os.path.join(path, file)))
    
    # Check if data is not empty
    if not wt_data or not mutant_data or not phage_data or not time_data:
        print("No data found.")
        return
    
    # Create figure and axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # colors = ["red", "orange", "yellow", "green", "blue", "lightblue", "violet", "grey", "black", "cyan", "lightgreen", "purple", ""]
    import matplotlib
    colors = list(matplotlib.colors.get_named_colors_mapping())
    np.random.shuffle(colors)
    # Plot data
    for i in range(len(wt_data)):
        # print(wt_data[i].shape, len(time_data[i]))
        if len(wt_data) > 1:
            for j in range(len(wt_data[i])):
                # print(len(wt_data[i][j]), len(time_data[i]))
                if wt_data[i][j].sum() + mutant_data[i][j].sum() == 0:
                    continue
                ax1.plot(time_data[i], wt_data[i][j]+mutant_data[i][j], color=colors[j])
                # ax1.plot(time_data[i], mutant_data[i][j], '--', color=ecolors[j])
                ax2.plot(time_data[i], np.divide(mutant_data[i][j], (wt_data[i][j] + mutant_data[i][j]), out=np.zeros_like(mutant_data[i][j]), where=(wt_data[i][j] + mutant_data[i][j])!=0), color=colors[j])
                ax3.plot(time_data[i], np.log10(phage_data[i][j]))
        else:
            ax1.plot(time_data[i], wt_data[i])
            ax1.plot(time_data[i], mutant_data[i], '--',)
            ax2.plot(time_data[i], np.divide(mutant_data[i], (wt_data[i] + mutant_data[i]), out=np.zeros_like(mutant_data[i]), where=(wt_data[i] + mutant_data[i])!=0))
            ax3.plot(time_data[i], np.log10(phage_data[i]))

    
    # Set labels and legends
    ax1.set_ylabel('Bacterial Population')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_ylabel('Mutant Fraction')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_ylabel('Phage Population')
    ax3.legend()
    ax3.grid(True)
    
    # Show or save plot
    if show:
        plt.tight_layout()
        plt.show()
    if save:
        plt.tight_layout()
        plt.savefig(save_path)


# def visualize_all_populations(history_wt_path, history_mutant_path, history_phage_path, time_path, path, save=False, save_path=None, show=True):
    
#     # Load data
#     wt_data = np.loadtxt(history_wt_path)
#     mutant_data = np.loadtxt(history_mutant_path)
#     phage_data = np.loadtxt(history_phage_path)
#     time_data = np.loadtxt(time_path)
    
#     # Handle single chemostat case
#     if wt_data.ndim == 1:
#         wt_data = wt_data.reshape(-1, 1)
#         mutant_data = mutant_data.reshape(-1, 1)
#         phage_data = phage_data.reshape(-1, 1)
    
#     # Calculate fraction of mutants
#     total_bacteria = wt_data + mutant_data
#     print(total_bacteria[-1][-1], phage_data[-1][-1])
#     mutant_fraction = np.divide(mutant_data, total_bacteria, out=np.zeros_like(mutant_data), where=total_bacteria!=0)
#     if wt_data.shape[1] == 1:
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#         ax1.plot(time_data, wt_data, label=f'WT Chemostat')
#         ax1.plot(time_data, mutant_data, '--', label=f'Mutant Chemostat')
#         ax1.set_ylabel('Bacterial Population')
#         ax1.legend()
#         ax1.grid(True)
        
#         ax2.plot(time_data, mutant_fraction, label=f'Mutant Fraction')
#         ax2.set_ylabel('Mutant Fraction')
#         ax2.legend()
#         ax2.grid(True)
        
#         ax3.plot(time_data, np.log10(phage_data), label=f'Phage Population')
#         ax3.set_ylabel('Phage Population')
#         ax3.legend()
#         ax3.grid(True)
        
#         plt.tight_layout()
#         plt.show()
        
#     else:
#         for i in range(wt_data.shape[0]):

#             fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
#             # Plot bacterial populations (WT and mutant)
#             ax1.plot(time_data, wt_data[i, :], label=f'WT Chemostat {i+1}')
#             ax1.plot(time_data, mutant_data[i, :], '--', label=f'Mutant Chemostat {i+1}')
#             ax1.set_ylabel('Bacterial Population')
#             ax1.legend()
#             ax1.grid(True)
            
#             # Plot mutant fraction
#             ax2.plot(time_data, mutant_fraction[i, :], label=f'Chemostat {i+1}')
#             ax2.set_ylabel('Mutant Fraction')
#             ax2.legend()
#             ax2.grid(True)
            
#             # Plot phage populations
#             ax3.plot(time_data, phage_data[i, :], label=f'Chemostat {i+1}')
#             ax3.set_xlabel('Time')
#             ax3.set_ylabel('Phage Population')
#             ax3.legend()
#             ax3.grid(True)
            
#             plt.tight_layout()
#             plt.show()
    # plt.figure()    
    # with open(f"{path}/final_state_{0}.txt", "r") as fl:
    #     mtx = []
    #     for line in fl.readlines():
    #         mtx.append(list(map(float, line.strip().split())))
    #     matrix = np.array(mtx)
    # plt.imshow(np.log(matrix), aspect='auto')
    # plt.show()

    # plt.figure()
    # with open(f"{path}/final_state_mutant_{0}.txt", "r") as fl:
    #     mtx = []
    #     for line in fl.readlines():
    #         mtx.append(list(map(float, line.strip().split())))
    #     matrix = np.array(mtx)
    # plt.imshow(np.log(matrix), aspect='auto')
    # plt.show()



# Example usage with the current path
visualize_all_populations(path=path)



# cheater rare N=282_679.34955911565, Phage=56_316_858.31866364
# cheater abundante N+52_047.049214934654 Phage=160_613_364.01576668

# python3 MasterEquationSimulationPCDStructured.py -A 28.0 -B 0.05 -C 1e-08 -D 0.0 --phage_influx 100000000.0 -E 0.0125 -F 1200.0 -dft 1.0 -dfc 1.0 -a 0.9473684211 -a_mut 0 -r 0.004605 -r_mut 0.004605 --discretization_damage 1001 --discretization_volume 41 --nondivision_threshold 1000 --debug --save --max_time 3600 --number_of_chemostats 2 --max_delta_t_limit 0.005 -ni 1000000
# Nondivision_threshold=1000 a=0.9473684211, N=22_042_881, Phage=14_558_098
# python3 MasterEquationSimulationPCDStructured.py -A 28.0 -B 0.05 -C 1e-08 -D 0.0 --phage_influx 100000000.0 -E 0.0125 -F 1200.0 -dft 1.0 -dfc 1.0 -a 0.9473684211 -a_mut 0 -r 0.004605 -r_mut 0.004605 --discretization_damage 1001 --discretization_volume 41 --nondivision_threshold 1 --debug --save --max_time 3600 --number_of_chemostats 2 --max_delta_t_limit 0.005 -ni 1000000
# Nondivision_threshold=1 a=0.9473684211, N=22_052_683, Phage=14_535_890
# python3 MasterEquationSimulationPCDStructured.py -A 28.0 -B 0.05 -C 1e-08 -D 0.0 --phage_influx 100000000.0 -E 0.0125 -F 1200.0 -dft 1.0 -dfc 1.0 -a 0.9473684211 -a_mut 0 -r 0.004605 -r_mut 0.004605 --discretization_damage 1001 --discretization_volume 41 --nondivision_threshold 1000 --debug --save --max_time 3600 --number_of_chemostats 2 --max_delta_t_limit 0.005 -ni 1000000
# Mutant rises up in frequency from 50% to 56.5%. Which is higher than for Nondivision_threshold=1 (around 52%)
# Or from 40% to 47%. In both cases we start with super low population size.
# python3 MasterEquationSimulationPCDStructured.py -A 28.0 -B 0.05 -C 1e-08 -D 0.0 --phage_influx 100000000.0 -E 0.0125 -F 1200.0 -dft 1.0 -dfc 1.0 -a 0.95 -a_mut 0 -r 0.0 -r_mut 0.0 --discretization_damage 1001 --discretization_volume 41 --nondivision_threshold 1000 --debug --save --max_time 3600 --number_of_chemostats 2 --max_delta_t_limit 0.005 -ni 1000000
# Mutant rises up in frequency from 50% to ~ 77% (maybe then goes down, not finished), but the population dies out (because no immunity). 0.01 - 0.07 % of altruists commit suicide every step.

def plot_matrix_with_numbers(matrix, title=None, threshold=None, fmt='.2e'):
    """
    Plot a matrix as a grid with numbers inside squares.
    Optimized for large matrices (e.g., 40x1000).
    
    Args:
        matrix: 2D numpy array or list of lists
        title: Optional string for plot title
    """
    # Convert to numpy array if not already
    matrix = np.array(matrix)
    matrix /= matrix.sum()
    if threshold:
        threshold = matrix.max()*0.01
    
    matrix[matrix == 0] = np.nan
    non_nan_cols = ~np.all(np.isnan(matrix), axis=0)
    last_valid_col = np.max(np.where(non_nan_cols)[0])
    matrix = matrix[:, :last_valid_col+1]
    if threshold:
        last_valid_col = np.max(np.where(matrix > threshold)[1])
        matrix = matrix[:, :last_valid_col+1]
    rows, cols = matrix.shape
    
    plt.figure(figsize=(20, 8))    
    plt.imshow(matrix, aspect='auto', cmap='viridis')    
    plt.colorbar(label='Value')
    for i in range(rows):
        for j in range(cols):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if threshold is None or val > threshold:
                text = f'{val:{fmt}}'
                color = 'white' if val/matrix[~np.isnan(matrix)].max() < 0.6 else 'black'
                plt.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=6)
    
    if title:
        plt.title(title)    
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    plt.show()

# with open(f"{path}/final_state_{0}.txt", "r") as fl:
#         mtx = []
#         for line in fl.readlines():
#             mtx.append(list(map(float, line.strip().split())))
#         matrix = np.array(mtx)
# plot_matrix_with_numbers(matrix, "Example Matrix")

# with open(f"{path}/final_state_mutant_{0}.txt", "r") as fl:
#         mtx = []
#         for line in fl.readlines():
#             mtx.append(list(map(float, line.strip().split())))
#         matrix = np.array(mtx)
# plot_matrix_with_numbers(matrix, "Example Matrix", threshold=matrix.max()*0.01)


