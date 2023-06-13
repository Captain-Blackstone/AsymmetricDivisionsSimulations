from pathlib import Path

import pandas as pd
import sys

filename = sys.argv[1]


colnames = ["A", "B", "C", "D", "E", "F", "G", "a", "r",
            "phis", "population_sizes", "deltas", "errors",
            "p_size", "q_size", "time", "converged"]
rows = []
for folder in Path("data").glob("*"):
    params = list(map(float, folder.stem.split("_")))

    with (folder/Path("history.txt")).open("r") as fl:
        phis = list(map(float, fl.readline().strip().split()))
        population_sizes = list(map(float, fl.readline().strip().split()))
        deltas = list(map(float, fl.readline().strip().split()))[1:]
        errors = list(map(float, fl.readline().strip().split()))

    with (folder / Path("meta.txt")).open("r") as fl:
        p_array_size = int(fl.readline().strip().split()[-1])
        q_array_size = int(fl.readline().strip().split()[-1])
        time = float(fl.readline().strip().split()[-1])
        convergence = bool(fl.readline().strip().split()[-1])
    row = params + [phis, population_sizes, deltas, errors, p_array_size, q_array_size, time, convergence]
    rows.append(row)


df = pd.DataFrame(rows)
df.columns = colnames
df.to_csv(filename, sep="\t", index=False)
