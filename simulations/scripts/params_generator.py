import numpy as np
params = dict(
    asymmetry=np.linspace(0, 1, 6),
    repair=np.linspace(0, 1, 6),
    A=np.linspace(1e-4, 5, 5),
    B=np.linspace(1e-4, 0.5, 5),
    C=[1e-4, 1e-5],
    D=np.linspace(0, 1, 5),
    E=np.linspace(0, 1, 5)
)

x = 1
nu = 1
V = 1000
K = 1
i = 0


commands = []
allow = False
for a in params["asymmetry"]:
    for E in params["E"]:
        if E == 0:
            continue
        for A in params["A"]:
            for B in params["B"]:
                for C in params["C"]:
                    for D in params["D"]:
                        for r in np.linspace(0, E, 6):
                            "0.6000000000000001_0.05_5.0_0.125075_1e-05_0.5_0_0.25.txt"
                            a = 0.6
                            r = 0.05
                            A = 5
                            B = 0.125075
                            C = 1e-05
                            D = 0.5
                            F = 0
                            E = 0.25
                            i += 1
                            phi0 = A / x / nu
                            dilution_rate = B * V
                            nutrient_critical_amount = C * V / x / nu
                            damage_accumulation_rate = D * K
                            command = f"~/myenv/bin/python3 ChemostatSimulation.py -dalc {damage_accumulation_rate} " \
                                      f"--mode cluster -nar {nu} -mr {phi0} -nvsf {x} " \
                                      f"--dilution_rate {dilution_rate} -v " \
                                      f"{V} -nca {nutrient_critical_amount} " \
                                      f"-ni 100000000000000000 " \
                                      f"--deterministic_threshold 0 " \
                                      f"-a {a} -dri {r} --max_repair {E} --smart_initialization"
                            if allow:
                                allow = False
                            if command == "~/myenv/bin/python3 ChemostatSimulation.py -dalc 0.25 --mode cluster -nar 1 -mr 1.250075 -nvsf 1 --dilution_rate 0.1 -v 1000 -nca 0.1 -ni 100000000000000000 --deterministic_threshold 0 -a 0.0 -dri 0.15 --max_repair 0.25 --smart_initialization":
                                allow = True

                            print(command)
                            if a == 0.0:
                                print(command)
                            # commands.append(command)
print(i)
# with open("commands_for_cluster.txt", "w") as fl:
#     fl.write("\n".join(commands))