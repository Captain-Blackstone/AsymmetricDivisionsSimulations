from pathlib import Path

import numpy as np

lines = []
for file in Path("equilibria/").glob("*"):
    params = list(map(float, file.stem.split("_")))
    with file.open("r") as fl:
        file_lines = fl.readlines()
        phi = float(file_lines[0].strip())
        rhos = "\t".join(file_lines[1].strip().split())
        popsize = float(file_lines[2].strip())
        params.extend([popsize, phi, rhos])
    with file.open("r") as fl:
        if "overtime" in fl.read():
            params.append(False)
        else:
            params.append(True)
    lines.append("\t".join(list(map(str, params))))
header = "asymmetry\trepair\tA\tB\tC\tD\tE\tF\tN\tphi\tfinished"
header += "\t".join(list(map(lambda el: str(round(el, 5)), np.linspace(0, 1, len(file_lines[1].strip().split())))))
with open("parameter_search.tsv", "w") as fl:
    fl.write(header + '\n')
    fl.write("\n".join(lines))
