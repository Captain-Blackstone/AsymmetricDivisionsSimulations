import numpy as np
params = dict(
    asymmetry=np.linspace(0, 1, 6),
    repair=np.linspace(0, 1, 6),
    A=[692],
    B=np.linspace(1e-4, 0.5, 6),
    C=[1e-6, 1e-5, 1e-4],
    D=list(np.linspace(0, 0.25, 5)) + list(np.linspace(0.5, 1, 3)),
    E=list(np.linspace(0, 0.25, 5)) + list(np.linspace(0.5, 1, 3)),
    F=[1],
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
                                for r in np.linspace(0, E, 6):
                                    i += 1
                                    command = f"~/myenv/bin/python3 ChemostatSimulation.py --mode cluster " \
                                              f"-A {A} -B {B} -C {C} -D {D} -E {E} -F {F} -G {G} -a {a} -r {r} " \
                                              f"-ni 100000000000000000 "
                                    commands.append(command)
                                    # print(command)
print(i)
with open("commands_for_cluster.txt", "w") as fl:
    fl.write("\n".join(commands))