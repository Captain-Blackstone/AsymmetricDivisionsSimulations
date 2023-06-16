import numpy as np
from pathlib import Path
params = dict(
    asymmetry=np.linspace(0, 1, 6),
    repair=np.linspace(0, 1, 6),
    A=[28],
    B=np.linspace(0.1, 0.5, 6),
    C=np.linspace(1e-6, 1e-4, 6),
    D=np.linspace(0, 0.15, 6),
    E=np.linspace(0, 1, 6)[1:],
    F=[0],
    G=[1]
)

print("A", params["A"])
print("B", params["B"])
print("C", params["C"])
print("D", params["D"])
print("E", params["E"])
print("F", params["F"])
print("G", params["G"])


commands = []
i = 0
Path("runs").mkdir(exist_ok=True)
for a in params["asymmetry"]:
    for E in params["E"]:
        if E == 0:
            continue
        for A in params["A"]:
            for B in params["B"]:
                for C in params["C"]:
                    for D in params["D"]:
                        for F in params["F"]:
                            for G in params["G"]:
                                for r in np.linspace(0, D, 6):
                                    if r >= E:
                                        continue
                                    i += 1
                                    command = f"~/myenv/bin/python3 MasterEquationSolver.py --mode cluster " \
                                              f"-A {A} -B {B} -C {C} -D {D} -E {E} -F {F} -G {G} -a {a} -r {r} " \
                                              f"--discretization_volume 41 --discretization_damage 41"
                                    text = f"#!/bin/bash\n{command}"
                                    with open(f"runs/run_{i}.sh", "w") as fl:
                                        fl.write(text)
                                    commands.append(command)
                                    # print(command)
print(i)
with open("commands_for_cluster.txt", "w") as fl:
    fl.write("\n".join(commands))




